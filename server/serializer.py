def face_serializer(timestamp, person_id, camera_id, image_path,confidence) -> dict:
    return {"timestamp": timestamp,
            "personId": person_id,
            "cameraId": camera_id,
            "image": image_path,
            "confidence":confidence}
