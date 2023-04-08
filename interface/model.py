from roboflow import Roboflow

def load_model(model_name="plate_detect"):
    if model_name == "plate_detect":
        rf = Roboflow(api_key="dyX99EJ9Yo09I3Dj5aDm")
        project = rf.workspace().project("license-plates-recognition-pntau")
        model = project.version(2).model
        return model