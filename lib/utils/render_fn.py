""" from Yuting Ge 
    render rgb depth normal map with pytorch3d
    2024.05.02
"""

from typing import Any, List, Tuple, Union, Dict

import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (BlendParams, Materials, MeshRasterizer,
                                MeshRenderer, RasterizationSettings,
                                TexturesUV, TexturesVertex)
from pytorch3d.renderer.cameras import (CamerasBase, OrthographicCameras,
                                        PerspectiveCameras,
                                        look_at_view_transform)
from pytorch3d.renderer.lighting import (AmbientLights, DirectionalLights,
                                         PointLights)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shader import HardFlatShader, HardPhongShader
from pytorch3d.structures import Meshes
from torch import Tensor

Device = Union[str, torch.device]


## ------------------
def normalize_verts_withBbox(verts:torch.Tensor):
    """
    Parameters
    ----
    verts : Tensor 
        `[B, n_verts, 3]`
        
    Returns
    ----
    verts : Tensor 
        `[B, n_verts, 3]`, with XY in [-1, 1] and Z in (0, 2].

    """
    b, n, k = verts.shape

    max_coor = verts.reshape(-1, k).max(dim=0, keepdim=True).values # [3,]
    min_coor = verts.reshape(-1, k).min(dim=0, keepdim=True).values # [3,]

    center = (max_coor + min_coor) / 2
    scale  = (max_coor - min_coor).max() / 2

    # to promise the Z in (0, 1) to meet the renderer settings.
    normed_verts = torch.zeros_like(verts).to(device=verts.device)
    normed_verts[..., 2] = (verts[..., 2] - center[..., 2]) / (scale + 1e-5) + 1 + 1e-5
    normed_verts[...,:2] = (verts[...,:2] - center[...,:2]) / (scale + 1e-5)

    return normed_verts



## ----------------------------------
class ColorsArgs(dict):
    """
    Inputs: 3 element tensor/tuple/list (different objects the same settings) or 
            list of lists (different objects the differents settings). [[a,b,c],]
    """ 
    ambient_color: Union[Tuple, Tensor]
    diffuse_color: Union[Tuple, Tensor]
    specular_color: Union[Tuple, Tensor]

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_tensor(self, device:Device, dtype=torch.float32):
        for key in ['ambient_color', 'diffuse_color', 'specular_color']:
            val = getattr(self, key, None)
            if val is not None:
                val = torch.tensor(val, dtype=dtype, device=device)
                setattr(self, key, val)
        return self
        

## ---------------------------------
"""Predefined Hand Materials."""
HAND_MATERIAL = ColorsArgs(ambient_color =torch.tensor([[0.95, 0.95, 0.95],]),
                           diffuse_color =torch.tensor([[0.85, 0.85, 0.85],]),
                           specular_color=torch.tensor([[0.00, 0.00, 0.00],]))


## =================================
class Pytorch3DRenderer(torch.nn.Module):

    def __init__(self,
        image_size:tuple=(512, 512),
        smooth_skinning:bool=True, # if rendering smooth skinning surface or not
        materials_args:ColorsArgs={},
        face_per_pixel:int=1, # zj: allow generate the back depth map.
        device: Device='cpu',
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = device
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)

        # rasterizer
        rasterSettings = RasterizationSettings(image_size, blur_radius=0., 
                                               cull_backfaces=False, bin_size=0, faces_per_pixel=face_per_pixel) # zj
        self.rasterizer = MeshRasterizer(raster_settings=rasterSettings).to(device=self.device)

        # shader
        # 'Hard' means only assign the closest face for each pixel while 'Soft' will consider all the faces project in one pixel.
        self.smooth_skinning = smooth_skinning
        blendParams = BlendParams(sigma=1e5, gamma=1e-5, background_color=(0,0,0))
        _Shader = HardPhongShader if smooth_skinning else HardFlatShader 
        self.shader = _Shader(blend_params=blendParams, device=self.device)
        
        # render
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader).to(device=self.device)

        # default marerials and lights
        self.lights = PointLights(device=self.device, location=((0, 0, -1.0),))
        self.materials = Materials(device=self.device, **materials_args)

        # default cameras
        # note: In Pytorch3D camera coordinates +X left, +Y up, +Z into the image plane, while in OpenCV camera that always +X right, +Y down, +Z into the plane. So here should have a fixing rotation for camera to produce common images.
        self.fix_R, T = look_at_view_transform(dist=1.0, elev=0., azim=180., up=((0, -1, 0),))
        self.default_cameras = dict(type='ortho', cam=OrthographicCameras(device=self.device, R=self.fix_R, T=T))


    # --------
    def to(self, device:Device):

        self.rasterizer = self.rasterizer.to(device=device)
        self.shader = self.shader.to(device=device)
        self.renderer = self.renderer.to(device=device)
        self.lights = self.lights.to(device=device)
        self.materials = self.materials.to(device=device)
        self.device = device

        return self


    # --------
    def set_materials(self, materials_args:ColorsArgs={}, shinines:int=64):
        self.materials = Materials(device=self.device, shininess=shinines, **materials_args)
        

    # --------
    def create_perspective_cameras(self, K:Tensor, R:Tensor=None, T:Tensor=None) -> Dict:
        """
        Create perspective cameras for projection.

        Parameters
        ----
        K : Tensor
            `[N, 4, 4]`. The intrinsics of camera.
        R : Tensor
            `[N, 3, 3]`. The rotation of camera, World -> Camera. 'None' for Identity.
        T : Tensor
            `[N, 3]`. The translation of camera, World -> Camera. 'None' for zeros.

        Returns
        ----
        cameras : CamerasBase
            The cameras instance for projection in Pytorch3DRenderer.

        .. code-block:: python            
            K = [[fx,   0,   px,   0],
                 [0,   fy,   py,   0],
                 [0,    0,    0,   1],
                 [0,    0,    1,   0],]
        """
        assert K.ndim == 3, ValueError(f'Expected K shape in [N, 4, 4], but get {K.shape}')

        N = K.shape[0] # batch size
        
        if R is not None:
            assert R.shape[0] == N
        if T is not None:
            assert T.shape[0] == N

        cam_R = self.fix_R if R is None else (self.fix_R @ R)
        cam_T = torch.zeros([N, 3]) if T is None else T
        camera = PerspectiveCameras(K=K.clone(), R=cam_R, T=cam_T, 
                                    image_size=[self.image_size]*N, 
                                    in_ndc=False, device=self.device)

        return dict(type='persp', cam=camera)


    # --------
    def create_orthographic_cameras(self, K:Tensor=None, R:Tensor=None, T:Tensor=None):
        """
        Create orthographic cameras for projection.

        Parameters
        ----
        K : Tensor
            `[N, 4, 4]`. The intrinsics of orthographic camera.
        R : Tensor
            `[N, 3, 3]`. The rotation of camera, World -> Camera. 'None' for Identity.
        T : Tensor
            `[N, 3]`. The translation of camera, World -> Camera. 'None' for zeros.

        Returns
        ----
        cameras : CamerasBase
            The cameras instance for projection in Pytorch3DRenderer.

        .. code-block:: python            
            K = [
                    [fx,   0,    0,  px],
                    [0,   fy,    0,  py],
                    [0,    0,    1,   0],
                    [0,    0,    0,   1],
            ]

            # fx is actually scale_x, same as fy
        """
        assert K.ndim == 3, ValueError(f'Expected K shape in [N, 4, 4], but get {K.shape}')

        if K is None:
            return self.default_cameras
        
        N = K.shape[0] # batch size

        if R is not None:
            assert R.shape[0] == N
        if T is not None:
            assert T.shape[0] == N

        cam_R = self.fix_R if R is None else (self.fix_R @ R)
        cam_T = torch.zeros([N, 3]) if T is None else T
        camera = OrthographicCameras(K=K.clone(), R=cam_R, T=cam_T, 
                                     image_size=[self.image_size]*N, 
                                     in_ndc=False, device=self.device)

        return dict(type='ortho', cam=camera)


    # --------
    def adjust_vertices_with_cam(self, vertices:Tensor, cam_type:str='persp'):
        if cam_type == 'persp':
            return vertices
        elif cam_type == 'ortho': # for ortho-camera, the z values should > 0., or will be crop off.
            vertices[..., 2] = vertices[..., 2] - vertices[..., 2].min().detach() + 1e-5
            return vertices
        else:
            raise ValueError('Invalid camera type')


    # --------
    def get_renderMap_UV(self, 
                         vertices:Tensor, 
                         faces:Tensor, 
                         texUV:TexturesUV, 
                         cameras:Dict=None, 
                         **kwargs) -> Tensor:
        """ 
        Rendering RGBA images with TextureUV.

        Parameters
        ----
        vertices : Tensor
            `[N, n_verts, 3]`.
        faces : Tensor
            `[N, n_faces, 3]`.
        texUV : TexturesUV
            UVs of current mesh.
        cameras : CamerasBase
            Cameras of current scene. If given 'None', will defaultly be a standard camera with focal length 1 and offset 0 in NDC space.
        lights : Optional 
            Instance of Pytorch3D's lights classes.
        materials : Optional 
            Instance of Pytorch3D's materials classes. 
        
        Returns
        ----
        renderMap : Tensor
            `[N, H, W, 4]`. output of RGBA images.

        Examples
        ----
        >>> renderer = Pytorch3DRenderer(...)
            texture_map = torch.tensor(cv2.imread(...)) / 255.
            texture_uv  = TexturesUV(tex_map, faces_uv, verts_uv)
            rgba = renderer.get_renderMap_UV(v, f, texture_uv)
        
        """
        
        vertices, faces = vertices.to(self.device), faces.to(self.device, dtype=torch.long)
        assert isinstance(texUV, TexturesUV), 'Texture should be the instance of \'TexturesUV\'.'

        cameras = self.default_cameras if cameras is None else cameras
        vertices = self.adjust_vertices_with_cam(vertices, cameras['type'])

        lights = kwargs.get('lights', None)
        lights = self.lights if lights is None else lights
        lights.to(device=self.device)

        materials = kwargs.get('materials', None)
        materials = self.materials if materials is None else materials
        materials.to(device=self.device)

        meshes = Meshes(verts=vertices, faces=faces, textures=texUV).to(self.device)     
        renderMap = self.renderer(meshes, cameras=cameras['cam'].to(device=self.device), 
                                  lights=lights, materials=materials)

        return renderMap # RGBA


    # --------
    def get_renderMap(self, 
                      vertices:Tensor, 
                      faces:Tensor, 
                      textures:Tensor=None, 
                      cameras:CamerasBase=None, 
                      **kwargs) -> Tensor:
        """ 
        Rendering RGBA images.

        Parameters
        ----
        vertices : Tensor
            `[N, n_verts, 3]`.
        faces : LongTensor
            `[N, n_faces, 3]`.
        textures : Tensor
            `[N, n_verts, 3]`, texture of each vert.
        cameras : CamerasBase
            Cameras of current scene.
        lights : Optional 
            Instance of Pytorch3D's lights classes.
        materials : Optional 
            Instance of Pytorch3D's materials classes. 
        
        Returns
        ----
        renderMap : Tensor
            `[N, H, W, 4]`. output of RGBA images.
        
        """
        
        vertices, faces = vertices.to(self.device), faces.to(self.device, dtype=torch.long)
        if textures is not None:
            textures = TexturesVertex(textures.to(device=self.device))
        else:
            textures = TexturesVertex(torch.ones_like(vertices).to(device=self.device) * 0.8)

        cameras = self.default_cameras if cameras is None else cameras
        vertices = self.adjust_vertices_with_cam(vertices, cameras['type'])

        lights = kwargs.get('lights', None)
        lights = self.lights if lights is None else lights
        lights.to(device=self.device)

        materials = kwargs.get('materials', None)
        materials = self.materials if materials is None else materials
        materials.to(device=self.device)

        meshes = Meshes(verts=vertices, faces=faces, textures=textures).to(self.device)     
        renderMap = self.renderer(meshes, cameras=cameras['cam'].to(device=self.device), 
                                  lights=lights, materials=materials)

        return renderMap # RGBA


    # --------
    def get_normalMap(self, 
                      vertices:Tensor, 
                      faces:Tensor, 
                      cameras:CamerasBase=None, 
                      **kwargs) -> Tensor:
        """ 
        Output normal maps. Refering to: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/mesh/shading.py

        Parameters
        ----
        vertices : Tensor
            `[N, n_verts, 3]`.
        faces : Tensor
            `[N, n_faces, 3]`.
        cameras : CamerasBase
            Cameras of current scene.

        Returns
        ----
        normalMap : Tensor
            `[N, H, W, 3]`.

        """

        vertices, faces = vertices.to(self.device), faces.to(self.device, dtype=torch.long)
        cameras = self.default_cameras if cameras is None else cameras
        vertices = self.adjust_vertices_with_cam(vertices, cameras['type'])

        meshes = Meshes(verts=vertices, faces=faces).to(self.device) 
        rasterOut = self.rasterizer(meshes, 
                                    cameras=cameras['cam'].to(device=self.device))

        normalMaps = None
        if self.smooth_skinning:
            faces = meshes.faces_packed()  # (F, 3)
            vertex_normals = meshes.verts_normals_packed()  # (V, 3)
            faces_normals = vertex_normals[faces]
            normalMaps = interpolate_face_attributes(rasterOut.pix_to_face, rasterOut.bary_coords, faces_normals)
            normalMaps[rasterOut.pix_to_face == -1] = -1. # for black ground, (-1 + 1) / 2 = 0
        else:
            face_normals = meshes.faces_normals_packed()  # (V, 3)
            mask = rasterOut.pix_to_face == -1
            pix_to_face = rasterOut.pix_to_face.clone()
            pix_to_face[mask] = 0
            N, H, W, K = pix_to_face.shape
            idx = pix_to_face.view(N * H * W * K, 1).expand(N * H * W * K, 3)
            normalMaps = face_normals.gather(0, idx).view(N, H, W, K, 3)
            normalMaps[mask] = -1. # for black ground, (-1 + 1) / 2 = 0

        normalMaps = normalMaps[...,0,:] # only consider the nearest face of K faces in one pixel.
        return ((normalMaps + 1) / 2).clamp(0, 1)


    # ----------
    def get_depthMap(self, vertices:Tensor, faces:Tensor, cameras:CamerasBase=None,
    **kwargs) -> Tensor:
        """ 
        Output depth maps of camera space.
        
        Parameters
        ----
        vertices : Tensor
            `[N, n_verts, 3]`.
        faces : Tensor
            `[N, n_faces, 3]`.
        cameras : CamerasBase
            Cameras of current scene.

        Returns
        ----
        depthMap : Tensor
            `[N, H, W]`.
        """

        vertices, faces = vertices.to(self.device), faces.to(self.device, dtype=torch.long)
        cameras = self.default_cameras if cameras is None else cameras
        vertices = self.adjust_vertices_with_cam(vertices, cameras['type'])

        meshes = Meshes(verts=vertices, faces=faces).to(self.device)  
        rasterOut = self.rasterizer(meshes, 
                                    cameras=cameras['cam'].to(device=self.device))
        
        # region [zj]
        # return rasterOut.zbuf[...,0] # only consider the nearest face of K faces in one pixel.
        return rasterOut.zbuf, rasterOut.pix_to_face # TopK depth map & TopK face index.
        # endregion

    # ----------
    def get_rasterResults(self, vertices:Tensor, faces:Tensor, cameras:CamerasBase=None,
    **kwargs) -> Tensor:
        """ 
        Output raseter Results.
        
        Parameters
        ----
        vertices : Tensor
            `[N, n_verts, 3]`.
        faces : Tensor
            `[N, n_faces, 3]`.
        cameras : CamerasBase
            Cameras of current scene.

        Returns
        ----
        rasterResults : Dict
            format in {
                'pix_to_face': `[N, H, W, faces_per_pixel]`,
                'bary_coords': `[N, H, W, faces_per_pixel, 3]`,
                'dists': `[N, H, W, faces_per_pixel]`,
                'zbuf': `[N, H, W, faces_per_pixel]`,
            }
        pixel_to_face: The index of the face that is projected at the corresponding pixel.
        bary_coords: The barycentric coordinates of the pixel in the projected face.
        dists: The distance from the camera to the face.
        zbuf: The depth of the face at the pixel.
        """

        vertices, faces = vertices.to(self.device), faces.to(self.device, dtype=torch.long)
        cameras = self.default_cameras if cameras is None else cameras
        vertices = self.adjust_vertices_with_cam(vertices, cameras['type'])

        meshes = Meshes(verts=vertices, faces=faces).to(self.device)  
        rasterOut :Fragments = self.rasterizer(meshes, 
                                               cameras=cameras['cam'].to(device=self.device))
        
        # process pix_to_face, every pixel contains the idx of covering face.
        pix_to_face = rasterOut.pix_to_face
        pix_to_face = pix_to_face - meshes.mesh_to_faces_packed_first_idx().reshape(-1, 1, 1, 1)
        pix_to_face[pix_to_face < 0] = -1

        rasterResults = dict(
            pix_to_face=pix_to_face,
            bary_coords=rasterOut.bary_coords,
            dists=rasterOut.dists,
            zbuf=rasterOut.zbuf,
        )

        return rasterResults



# ----------
def batch_project_K(points3d:Tensor, K:Tensor) -> Tensor:
    """Projecting 3D points to 2D plane with camera K in batch.

    Parameters
    ----
    points3d : Tensor
        `[B, N, 3]`
    K : Tensor
        `[B, 4, 4]`

    Returns
    ----
    points2d : Tensor
        `[B, N, 2]`
    
    .. code-block:: python            
    K = [[fx,   0,   px,   0],
         [0,   fy,   py,   0],
         [0,    0,    0,   1],
         [0,    0,    1,   0],]
    """

    points2d = points3d / (points3d[..., 2:] + 1e-8)
    points2d = points2d @ K[..., :3, :3].transpose(-1, -2)

    return points2d[..., :2]





if __name__ == '__main__':
    pass
    # image_shape = img.shape[:2]
    # renderer = Pytorch3DRenderer(image_size=image_shape)
    # cameras = renderer.create_perspective_cameras(K=K, R=R, T=T)
    # normalMap = renderer.get_normalMap(hand_verts, hand_faces, cameras)
