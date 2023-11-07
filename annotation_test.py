import os
import numpy as np
import napari
import zarr
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url
import tensorstore as ts
import pathlib
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



zarr_path = r'D:\\Kasia\tracking\E6_exp\todel.zarr'

reader = Reader(parse_url(zarr_path))

image_node = list(reader())[0]
dask_data = image_node.data

viewer = napari.Viewer()
viewer.add_image(dask_data,name=napari.__version__,contrast_limits=[0,10000])


labels_test_path = r'D:\kasia\tracking\to_del_annotation.zarr'

def open_tensorstore(labels_file: pathlib.Path, *, shape=None, chunks=None):
    if not os.path.exists(labels_file):
        zarr.open(
                str(labels_file),
                mode='w',
                shape=shape,
                dtype=np.uint32,
                chunks=chunks,
                )
    # read some of the metadata for tensorstore driver from file
    labels_temp = zarr.open(str(labels_file), mode='r')
    metadata = {
            'dtype': labels_temp.dtype.str,
            'order': labels_temp.order,
            'shape': labels_temp.shape,
            }
    dir, name = os.path.split(labels_file)
    labels_ts_spec = {
            'driver': 'zarr',
            'kvstore': {
                    'driver': 'file',
                    'path': dir,
                    },
            'path': name,
            'metadata': metadata,
            }
    data = ts.open(labels_ts_spec, create=False, open=True).result()
    return data

persistent_seg = open_tensorstore(
    labels_test_path,
    shape=(100, 10000, 10000),
    chunks=(1,2048,2048)
)

viewer.add_labels(persistent_seg)


if __name__ == '__main__':
    napari.run()
