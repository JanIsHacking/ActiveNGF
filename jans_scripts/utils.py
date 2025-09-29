import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any


def parse_graspnet_scene(xml_path: str | Path) -> List[Dict[str, Any]]:
    """
    Parse a GraspNet XML scene file containing object poses.

    Args:
        xml_path: Path to the XML file.

    Returns:
        A list of dictionaries, one per object, with fields:
        - obj_id (int)
        - obj_name (str)
        - obj_path (str)
        - pos_in_world (list[float], length 3)
        - ori_in_world (list[float], length 4)
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("obj"):
        obj_id = int(obj.find("obj_id").text)
        obj_name = obj.find("obj_name").text.strip()
        obj_path = obj.find("obj_path").text.strip()

        pos_in_world = [float(x) for x in obj.find("pos_in_world").text.split()]
        ori_in_world = [float(x) for x in obj.find("ori_in_world").text.split()]
        ori_in_world = [ori_in_world[1], ori_in_world[2], ori_in_world[3], ori_in_world[0]]

        objects.append(
            {
                "obj_id": obj_id,
                "obj_name": obj_name,
                "obj_path": obj_path,
                "pos_in_world": pos_in_world,
                "ori_in_world": ori_in_world,
            }
        )

    return objects