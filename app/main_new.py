import base64
import io
import os
import time
import uuid
from typing import Dict

import cv2
import numpy as np
import requests  # Use requests to call the API
import trimesh
from fastapi import FastAPI, BackgroundTasks, HTTPException
from PIL import Image
from pydantic import BaseModel
from rembg import remove
from shapely.geometry import Polygon

# --- Configuration ---
# It's recommended to set your Gemini API Key as an environment variable
# for security reasons.
GEMINI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
# Switched to a dedicated image generation model compatible with the request format.
IMAGE_GENERATION_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent?key={GEMINI_API_KEY}"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- In-memory Job Tracking ---
jobs: Dict[str, Dict] = {}

# --- FastAPI App ---
app = FastAPI()


# --- Pydantic Models for API ---
# The Stable Diffusion specific parameters are removed.
class TerrainGenerationParams(BaseModel):
    prompt: str = "a majestic snowy mountain range, rocky, detailed texture, aerial view, grayscale"
    vertical_scale: float = 25.0


class PropGenerationParams(BaseModel):
    prompt: str = "a slim katana"
    extrusion_depth: float = 5.0


# --- New Gemini Image Generation Logic ---

def generate_image_with_gemini(prompt: str) -> Image.Image:
    """
    Generates an image using the Gemini API (gemini-2.5-flash-image-preview model).

    Args:
        prompt: The text prompt to generate an image from.

    Returns:
        A PIL Image object.

    Raises:
        HTTPException: If the API key is not set or if the API call fails.
    """
    if not GEMINI_API_KEY:
        # Added a more helpful message for the user.
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set. Please get a key from Google AI Studio and set it.")

    # Payload updated for the image generation model.
    # We specify that we expect an IMAGE in the response modalities.
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"]
        },
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(IMAGE_GENERATION_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()

        # Response parsing updated for the generateContent method's structure.
        candidate = response_data.get("candidates", [{}])[0]
        part = candidate.get("content", {}).get("parts", [{}])[0]

        if "inlineData" not in part:
            # Check for safety ratings or other issues if no image is returned.
            error_info = response_data.get("promptFeedback", "No specific error details provided by API.")
            raise ValueError(f"API response did not contain image data. Feedback: {error_info}")

        b64_string = part["inlineData"]["data"]
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image

    except requests.exceptions.RequestException as e:
        # Include the response text in the error for better debugging.
        error_detail = f"Failed to generate image from API. Status: {e.response.status_code}. Response: {e.response.text}"
        print(f"Error calling Gemini API: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error processing API response: {e}")
        raise HTTPException(status_code=500, detail=f"Invalid API response format: {e}")


# --- Updated Generation Logic ---

def generate_terrain(params: TerrainGenerationParams, job_id: str):
    """
    Worker function for generating terrain-like meshes from a heightmap.
    Uses pixel brightness for height.
    """
    try:
        jobs[job_id]["status"] = "generating_heightmap"
        print(f"Job {job_id}: Generating heightmap with Gemini...")

        # --- 1. Generate Heightmap Image using Gemini API ---
        image = generate_image_with_gemini(params.prompt)

        heightmap_filename = f"heightmap_{job_id}.png"
        heightmap_path = os.path.join(OUTPUT_DIR, heightmap_filename)
        # Convert to grayscale for heightmap processing
        image.convert("L").save(heightmap_path)
        jobs[job_id]["heightmap"] = heightmap_path
        print(f"Job {job_id}: Heightmap saved to {heightmap_path}")

        # --- 2. Convert Heightmap to Mesh (No changes here) ---
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
    Worker function for generating simple, solid props by extruding a 2D silhouette.
    """
    try:
        jobs[job_id]["status"] = "generating_image"
        print(f"Job {job_id}: Generating image for prop with Gemini...")

        # --- 1. Generate Base Image using Gemini API ---
        image = generate_image_with_gemini(params.prompt)

        image_filename = f"prop_image_{job_id}.png"
        image_path = os.path.join(OUTPUT_DIR, image_filename)
        image.save(image_path)
        jobs[job_id]["image"] = image_path
        print(f"Job {job_id}: Image saved to {image_path}")

        # --- 2. Remove Background to get Silhouette (No changes here) ---
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

        # --- 3. Convert Silhouette to 3D Mesh (No changes here) ---
        jobs[job_id]["status"] = "converting_to_mesh"
        print(f"Job {job_id}: Converting to mesh...")

        img = cv2.imread(bg_removed_path, cv2.IMREAD_UNCHANGED)
        # Ensure image has an alpha channel for contour finding
        if img.shape[2] != 4:
            raise ValueError("Image must have an alpha channel after background removal.")

        # Use the alpha channel as the grayscale image for finding contours
        alpha_channel = img[:, :, 3]
        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the image.")

        contour = max(contours, key=cv2.contourArea)

        if len(contour) < 3:
            raise ValueError(f"Contour is degenerate (has only {len(contour)} points).")

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

        if not all_meshes:
            raise ValueError("Mesh creation failed after cleaning the polygon.")

        mesh = trimesh.util.concatenate(all_meshes)

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Mesh creation failed, resulting in type {type(mesh)}.")

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


# --- API Endpoints ---
@app.post("/generate")
async def start_terrain_generation(params: TerrainGenerationParams, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "started_at": time.time(), "params": params.dict()}
    background_tasks.add_task(generate_terrain, params, job_id)
    return {"job_id": job_id}


@app.post("/generate/props")
async def start_prop_generation(params: PropGenerationParams, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "started_at": time.time(), "params": params.dict()}
    background_tasks.add_task(generate_prop, params, job_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/")
async def root():
    return {"message": "Welcome to the 3D Terrain and Prop Generator API (with Gemini)!"}
