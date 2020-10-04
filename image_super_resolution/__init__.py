# check if dependencies are available
hard_dependencies = ('numpy','cv2','skimage','scipy')

missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError("image-super-resolution: Unable to import required dependencies: \n"+"\n".join(missing_dependencies))
else:
    print("#####################################################")
    print("image-super-resolution: All dependencies available !")
    print("#####################################################")
    print("")
del hard_dependencies, dependency, missing_dependencies

