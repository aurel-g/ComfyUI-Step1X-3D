import warnings

warnings.filterwarnings("ignore")
import os
import torch
import trimesh

from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline








def texture_pipeline(input_image_path, input_glb_path, save_glb_path):
    """
    The texture model, input image and glb generate textured glb
    """
    mesh = trimesh.load(input_glb_path)
    pipeline = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture")
    mesh = remove_degenerate_face(mesh)
    mesh = reduce_face(mesh)
    textured_mesh = pipeline(input_image_path, mesh, seed=2025)
    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    textured_mesh.export(save_glb_path)


class LoadStep1X3DGeometryModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "stepfun-ai/Step1X-3D/Step1X-3D-Geometry-1300m"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("geometry_model",)
    FUNCTION = "load_model"
    CATEGORY = "Step1X-3D"

    def load_model(self, model_path):
        geometry_model = model_path
        return (geometry_model,)


class LoadStep1X3DGeometryLabelModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "stepfun-ai/Step1X-3D/Step1X-3D-Geometry-Label-1300m"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("geometry_label_model",)
    FUNCTION = "load_model"
    CATEGORY = "Step1X-3D"

    def load_model(self, model_path):
        geometry_label_model = model_path
        return (geometry_label_model,)


class LoadStep1X3DTextureModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "stepfun-ai/Step1X-3D/Step1X-3D-Texture"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("texture_model",)
    FUNCTION = "load_model"
    CATEGORY = "Step1X-3D"

    def load_model(self, model_path):
        texture_model = model_path
        return (texture_model,)


class LoadInputImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "examples/images/000.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("input_image_path",)
    FUNCTION = "load_image"
    CATEGORY = "Step1X-3D"

    def load_image(self, image_path):
        input_image_path = image_path
        return (input_image_path,)


class 3DGeometryGeneration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "geometry_model": ("MODEL",),
                "input_image_path": ("IMAGE",),
                "guidance_scale": ("FLOAT", {"default": 7.5}),
                "num_inference_steps": ("INT", {"default": 50}),
                "seed": ("INT", {"default": 2025}),
            }
        }

    RETURN_TYPES = ("UNTEXTUREDMESH",)
    RETURN_NAMES = ("untextured_mesh",)
    FUNCTION = "geometry_generation"
    CATEGORY = "Step1X-3D"

    def geometry_generation(self, geometry_model, input_image_path, guidance_scale, num_inference_steps, seed):
        """
        The base geometry model, input image generate glb
        """
        pipeline = Step1X3DGeometryPipeline.from_pretrained(geometry_model).to("cuda")
    
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed)
        untextured_mesh = pipeline(input_image_path, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator)
    
        return (untextured_mesh,)


class 3DGeometryLabelGeneration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "geometry_label_model": ("MODEL",),
                "input_image_path": ("IMAGE",),
                "symmetry": ("STRING", {"default": "x"}),
                "edge_type": ("STRING", {"default": "sharp"}),
                "guidance_scale": ("FLOAT", {"default": 7.5}),
                "octree_resolution": ("INT", {"default": 384}),
                "max_facenum": ("INT", {"default": 400000}),
                "num_inference_steps": ("INT", {"default": 50}),
                "seed": ("INT", {"default": 2025}),
            }
        }

    RETURN_TYPES = ("LABELUNTEXTUREDMESH",)
    RETURN_NAMES = ("label_untextured_mesh",)
    FUNCTION = "geometry_label_generation"
    CATEGORY = "Step1X-3D"

    def geometry_label_generation(self, geometry_label_model, input_image_path, symmetry, edge_type, guidance_scale, octree_resolution, max_facenum, num_inference_steps, seed):
        """
        The label geometry model, support using label to control generation, input image generate glb
        """
        pipeline = Step1X3DGeometryPipeline.from_pretrained(geometry_label_model).to("cuda")
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed)

        label = {"symmetry": symmetry, "edge_type": edge_type}
    
        label_untextured_mesh = pipeline(
            input_image_path,
            label=label,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            max_facenum=max_facenum,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
    
        return (label_untextured_mesh,)

    
