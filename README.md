# blender-synthetic-ballast

## configure VS code develop environment

1. Install Blender >= 3.1, add blender executable file to PATH.
2. Open workspace `blender-synthetic-ballast.code-workspace` via VS Code.
3. Select blender's python executable file as workspace interpreter. Usually in `/PATH/TO/INSTALL/BLENDER/VERSION/python/bin`.
4. Install required python libraries by 
```
/PATH/TO/INSTALL/BLENDER/VERSION/python/bin/python -m ensurepip --default-pip
/PATH/TO/INSTALL/BLENDER/VERSION/python/bin/python -m pip install -r requirements.exe
```
If it doesn't work, please refer to https://packaging.python.org/en/latest/tutorials/installing-packages/.

5. Test your configuration by running either of the following command under the repository root directory:

```
blender -P ./scripts/addon_helper.py  # blender window will pop out

blender -P ./scripts/addon_helper.py  -b # blender will run in background
```

## add new material

All materials will be managed in `materials.blend`. To add your new material, open `materials.blend` and make sure `node wrangler` add-on is installed.

1. Create a new object (can be any like a sphere).
2. Go `shading` window, select the object created. Add new material (should be `Principled BSDF`).
3. Click on the `Principled BSDF` node, press `Ctrl + Shift + T` to select all material files downloaded.
4. Rename the material.
5. Load the material in `scripts/main.py`

## run the program

```sh
blender -P ./scripts/main_displacement.py [-b] # running in either mode is okay
```

<!-- Two methods to simulate fine grains are implemented currently. 

* The fine grains are presented by a mesh of `landscape`.
    ```
    blender -P ./scripts/main_landscape.py [-b] # running in either mode is okay
    ```
* The fine grains are presented by a particle system simulated by `molecular` scripts addon.
    ```
    blender -P ./scripts/main_molecular.py # must running with blender window popped
    blender tmp/molecular/tmp.blend  -P scripts/swap_particles.py [-b] # running in either mode is okay -->

## TODOs:

- [ ] gradation calibration
- [x] label generation support for method `landscape` (with and w/o overlapping)
- [ ] material surveying
- [x] hair particle system for method `landscape`
- [x] hair particle system tuning for method `landscape`
- [ ] fine tuning for landscape and rock generator configuration
- [ ] automated camera trajectories and animation rendering
- [ ] solve COLMAP reconstruction issue

## large file (glf) configuration

1. Install `git-glf` according to your operating system. For Ubuntu:
```
sudo apt-get install git-lfs
```
2. Go to the repository root directory and run `git install lfs`
3. Add extensions or files to be tracked as large files `git lfs track "*.exr"` for example
4. Add to `.gitattributes` by `git add .gitattributes`
5. `git add YOUR_FILE` and `git commit -m COMMENTS` and `git push`
