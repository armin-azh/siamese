def face_serializer(timestamp, person_id: str, camera_id, image_path) -> dict:
    return {"timestamp": timestamp,
            "personid": person_id,
            "camera": camera_id,
            "image": image_path}
