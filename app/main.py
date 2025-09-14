import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import trimesh
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Dict
import uuid
import time
from rembg import remove
import cv2
from shapely.geometry import Polygon
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
jobs: Dict[str, Dict] = {}
app = FastAPI()

class GenerationParams(BaseModel):
    prompt: str = "a majestic snowy mountain range, rocky, detailed texture, aerial view, grayscale"
    seed: int = 42
    steps: int = 25
    guidance_scale: float = 8.0
    vertical_scale: float = 25.0

class PropGenerationParams(BaseModel):
    prompt: str = "a slim katana"
    seed: int = 42
    steps: int = 50
    guidance_scale: float = 8.0
    extrusion_depth: float = 5.0

def generate_and_convert(params: GenerationParams, job_id: str):
    """
    The main worker function that runs in the background for terrain generation.
    """
    try:
        jobs[job_id]["status"] = "generating_heightmap"
        print(f"Job {job_id}: Generating heightmap...")
        pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        generator = torch.manual_seed(params.seed)
        image = pipeline(
            params.prompt,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
        ).images[0]

        heightmap_filename = f"heightmap_{job_id}.png"
        heightmap_path = os.path.join(OUTPUT_DIR, heightmap_filename)
        image.convert("L").save(heightmap_path)
        jobs[job_id]["heightmap"] = heightmap_path
        print(f"Job {job_id}: Heightmap saved to {heightmap_path}")

        jobs[job_id]["status"] = "converting_to_mesh"
        print(f"Job {job_id}: Converting to mesh...")

        heightmap_data = np.array(Image.open(heightmap_path))
        height, width = heightmap_data.shape
        x_coords = np.linspace(0, width - 1, width)
        z_coords = np.linspace(0, height - 1, height)
        xx, zz = np.meshgrid(x_coords, z_coords)
        yy = (heightmap_data / 255.0) * params.vertical_scale
        vertices = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)

        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                v1 = i * width + j
                v2 = v1 + 1
                v3 = (i + 1) * width + j
                v4 = v3 + 1
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
        mesh_filename = f"terrain_{job_id}.glb"
        mesh_path = os.path.join(OUTPUT_DIR, mesh_filename)
        print("Hello, world")
        mesh.export(mesh_path)
        jobs[job_id]["mesh"] = mesh_path
        print(f"Job {job_id}: Mesh saved to {mesh_path}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["finished_at"] = time.time()

    except Exception as e:
        print(f"Job {job_id}: Failed with error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

def generate_prop(params: PropGenerationParams, job_id: str):
    """
    The main worker function that runs in the background for prop generation.
    """
    try:
        jobs[job_id]["status"] = "generating_image"
        print(f"Job {job_id}: Generating image for prop...")

        print(f"Job {job_id}: Loading model...")
        pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        print(f"Job {job_id}: Model loaded.")
        generator = torch.manual_seed(params.seed)
        image = pipeline(
            params.prompt,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
        ).images[0]

        image_filename = f"prop_image_{job_id}.png"
        image_path = os.path.join(OUTPUT_DIR, image_filename)
        image.save(image_path)
        jobs[job_id]["image"] = image_path
        print(f"Job {job_id}: Image saved to {image_path}")

        jobs[job_id]["status"] = "removing_background"
        print(f"Job {job_id}: Removing background...")
        with open(image_path, "rb") as f:
            image_data = f.read()

        output_data = remove(image_data)
        bg_removed_filename = f"prop_bg_removed_{job_id}.png"
        bg_removed_path = os.path.join(OUTPUT_DIR, bg_removed_filename)
        with open(bg_removed_path, "wb") as f:
            f.write(output_data)
        jobs[job_id]["bg_removed_image"] = bg_removed_path
        print(f"Job {job_id}: Background removed image saved to {bg_removed_path}")

        jobs[job_id]["status"] = "converting_to_mesh"
        print(f"Job {job_id}: Converting to mesh...")

        img = cv2.imread(bg_removed_path, cv2.IMREAD_UNCHANGED)

        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the image.")

        contour = max(contours, key=cv2.contourArea)

        print("\n" + "="*60)
        print("DEBUGGING: CAPTURING CONTOUR DATA...")

        contour_filename = f"debug_contour_{job_id}.txt"
        contour_filepath = os.path.join(OUTPUT_DIR, contour_filename)
        np.savetxt(contour_filepath, contour.reshape(-1, 2), fmt='%d')

        print(f"Contour data has been saved to: {contour_filepath}")
        print(f"This contour has {len(contour)} points.")
        print("="*60 + "\n")
        if len(contour) < 3:
            raise ValueError(f"Contour is degenerate (has only {len(contour)} points) and cannot be extruded.")
        polygon_2d = Polygon(contour.reshape(-1, 2))

        cleaned_polygon = polygon_2d.buffer(0)
        all_meshes = []
        if cleaned_polygon.geom_type == 'Polygon':
            mesh = trimesh.creation.extrude_polygon(cleaned_polygon, height=params.extrusion_depth)
            all_meshes.append(mesh)
        elif cleaned_polygon.geom_type == 'MultiPolygon':
            for poly in cleaned_polygon.geoms:
                mesh = trimesh.creation.extrude_polygon(poly, height=params.extrusion_depth)
                all_meshes.append(mesh)
        if not isinstance(mesh, trimesh.Trimesh):
             raise TypeError(f"Mesh creation failed, resulting in type {type(mesh)} instead of a Trimesh object.")


        mesh_filename = f"prop_{job_id}.glb"
        mesh_path = os.path.join(OUTPUT_DIR, mesh_filename)
        mesh.export(mesh_path)
        jobs[job_id]["mesh"] = mesh_path
        print(f"Job {job_id}: Mesh saved to {mesh_path}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["finished_at"] = time.time()

    except Exception as e:
        print(f"Job {job_id}: Failed with error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/generate")
async def generate(params: GenerationParams, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "started_at": time.time(),
        "params": params.dict()
    }
    background_tasks.add_task(generate_and_convert, params, job_id)
    return {"job_id": job_id}

@app.post("/generate/props/")
async def generate_props(params: PropGenerationParams, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "started_at": time.time(),
        "params": params.dict()
    }
    background_tasks.add_task(generate_prop, params, job_id)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]

@app.get("/")
async def root():
    return {"message": "Welcome to the 3D Terrain and Prop Generator API!"}
