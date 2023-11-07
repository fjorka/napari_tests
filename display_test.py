import napari
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



zarr_path = './todel.zarr'

reader = Reader(parse_url(zarr_path))

image_node = list(reader())[0]
dask_data = image_node.data

viewer = napari.Viewer()
viewer.add_image(dask_data,name=napari.__version__,contrast_limits=[0,10000])


if __name__ == '__main__':
    napari.run()
