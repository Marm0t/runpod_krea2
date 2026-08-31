# Krea 2 + LoRA — RunPod Load Balancer

FastAPI worker для `krea/Krea-2-Turbo`. LoRA не входят в Docker image и читаются
только из подключенного RunPod Network Volume:

```text
/runpod-volume/lora/*.safetensors
```

Все найденные LoRA загружаются при старте worker. После изменения файлов на
volume worker нужно перезапустить.

## RunPod

Создайте Endpoint типа **Load Balancer**, подключите Network Volume и настройте:

| Переменная | Значение | Обязательна |
|---|---|---|
| `HF_TOKEN` | `hf_...` | Да |
| `LOAD_MODE` | `cpu_offload` для 24–32 GB VRAM, иначе `cuda` | Нет, default: `cuda` |
| `MAX_PIXELS` | `4194304` для изображений до 2048×2048 | Нет, default: `2097152` |
| `MODEL_ID` | `krea/Krea-2-Turbo` | Нет, это default |

`LOAD_MODE=cuda` быстрее, но требует больше VRAM. `cpu_offload` использует
sequential CPU offload для запуска на 24 GB VRAM и генерирует заметно медленнее.
Model cache:
`krea/Krea-2-Turbo`.

## API

Health check:

```http
GET /ping
```

Во время загрузки модели endpoint возвращает `204`, после полной готовности —
`200`, а при ошибке инициализации — `500`. Эти коды используются RunPod Load
Balancer для readiness: трафик генерации не должен попадать на холодный worker.

Список LoRA:

```http
GET /loras
```

Генерация:

```http
POST /generate
Content-Type: application/json
```

```json
{
  "prompt": "portrait photo of alice, cinematic light",
  "loras": [
    {"name": "alice", "scale": 1.0},
    {"name": "purelens_realism", "scale": 0.8}
  ],
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 8,
  "guidance_scale": 0.0,
  "seed": 42,
  "output_format": "jpeg",
  "quality": 92
}
```

Пустой `loras: []` означает генерацию чистой Krea 2 без LoRA. Если указанное имя
не найдено, `/generate` возвращает `400`. Порядок и веса массива передаются в
Diffusers без изменений.

## Проверка

```bash
source env.sh
python3 ping.py
python3 test_generate.py
```

```bash
python3 -m unittest discover -s tests -v
```
