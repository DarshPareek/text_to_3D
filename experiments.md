# Parameter Experiments

This document records the results of experiments with different input parameters for terrain generation.

## Experiment 1: The Effect of `seed`

The `seed` parameter controls the initial random state of the generation process. Changing the seed will produce a completely different terrain, even with the same prompt.

- **Seed 42:** A mountainous terrain with a prominent central peak.
- **Seed 123:** A more rolling hills landscape with a less defined central feature.
- **Seed 2023:** A canyon-like structure with a clear riverbed.

## Experiment 2: The Effect of `steps`

The `steps` parameter controls the number of inference steps the model takes. More steps can lead to a more detailed and refined output, but also increases generation time.

- **Steps 20:** The terrain is well-defined, but some areas lack fine detail.
- **Steps 30:** The terrain has more intricate details and fewer artifacts.

## Experiment 3: The Effect of `vertical_scale`

The `vertical_scale` parameter controls the height of the terrain. A higher value will result in more exaggerated mountains and valleys.

- **Vertical Scale 15:** A gentler landscape with rolling hills.
- **Vertical Scale 30:** A dramatic, exaggerated landscape with high peaks and deep valleys.
