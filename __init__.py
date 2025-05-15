from .nodes import LoadStep1X3DGeometryModel, LoadStep1X3DGeometryLabelModel, LoadStep1X3DTextureModel, LoadInputImage, GeometryGeneration, GeometryLabelGeneration, LoadUntexturedMesh, TexureSynthsis, SaveUntexturedMesh, SaveTexturedMesh

NODE_CLASS_MAPPINGS = {
    "LoadStep1X3DGeometryModel": LoadStep1X3DGeometryModel,
    "LoadStep1X3DGeometryLabelModel": LoadStep1X3DGeometryLabelModel,
    "LoadStep1X3DTextureModel": LoadStep1X3DTextureModel,
    "LoadInputImage": LoadInputImage,
    "GeometryGeneration": GeometryGeneration,
    "GeometryLabelGeneration": GeometryLabelGeneration,
    "LoadUntexturedMesh": LoadUntexturedMesh,
    "TexureSynthsis": TexureSynthsis,
    "SaveUntexturedMesh": SaveUntexturedMesh,
    "SaveTexturedMesh": SaveTexturedMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStep1X3DGeometryModel": "Load Step1X3D Geometry Model",
    "LoadStep1X3DGeometryLabelModel": "Load Step1X3D Geometry Label Model",
    "LoadStep1X3DTextureModel": "Load Step1X3D Texture Model",
    "LoadInputImage": "Load Input Image",
    "GeometryGeneration": "Geometry Generation",
    "GeometryLabelGeneration": "Geometry Label Generation",
    "LoadUntexturedMesh": "Load Untextured Mesh",
    "TexureSynthsis": "Texure Synthsis",
    "SaveUntexturedMesh": "Save Untexture dMesh",
    "SaveTexturedMesh": "Save Textured Mesh",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
