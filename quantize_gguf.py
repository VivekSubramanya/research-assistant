from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path('llama.cpp/gguf-py').resolve()))
import gguf
from gguf.quants import Q4_K

src = Path('models/qwen2.5-14b-research-assistant.f16.gguf')
out = Path('models/qwen2.5-14b-research-assistant.Q4_K_M.gguf')

reader = gguf.GGUFReader(str(src))
writer = gguf.GGUFWriter(str(out), arch='qwen2', endianess=reader.endianess)

writer.add_name('Qwen2.5 Research Assistant')
writer.add_description('Quantized via pure Python gguf tooling')
writer.add_file_type(15)

for field in reader.fields.values():
    if field.name in {gguf.Keys.General.ARCHITECTURE, 'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}:
        continue
    if field.name.startswith('GGUF.'):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()

for tensor in reader.tensors:
    data = tensor.data
    if data.dtype == np.float16 or data.dtype == np.float32:
        arr = data.astype(np.float32, copy=False)
        q = Q4_K.quantize(arr)
        writer.add_tensor(tensor.name, q.astype(np.uint8), raw_shape=tensor.shape, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    else:
        writer.add_tensor(tensor.name, data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)

writer.write_tensors_to_file(progress=False)
writer.close()
print(f'Wrote {out}')
