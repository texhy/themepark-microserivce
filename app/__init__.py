"""Theme Park face recognition microservice."""
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*rcond.*parameter will change.*",
    category=FutureWarning,
    module=r"insightface\.utils\.transform",
)
warnings.filterwarnings(
    "ignore",
    message=r".*estimate.*is deprecated.*",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align",
)

from app.core._nvidia_preload import preload as _nvidia_preload_cuda_libs

_nvidia_preload_cuda_libs()
del _nvidia_preload_cuda_libs
__version__ = "0.1.0"
