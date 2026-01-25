"""Exception classes for GPU Video Tools."""


class GPUVideoToolsError(Exception):
    """Base exception for GPU Video Tools."""
    pass


class DeviceMissingError(GPUVideoToolsError):
    """Raised when no device is specified in CLI, batch, or config."""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        message = (
            f"No device specified for tool '{tool_name}'.\n\n"
            f"Device must be specified via one of:\n"
            f"  1. CLI argument: --device <nvidia:0|amd:0|cpu|auto>\n"
            f"  2. Batch CSV column: 'device'\n"
            f"  3. Config file [defaults] section\n\n"
            f"To create an optimized default configuration, run:\n"
            f"  gpu-tools analyze --write-defaults\n\n"
            f"Or specify device explicitly:\n"
            f"  gpu-tools {tool_name} --device cpu <args>\n"
        )
        super().__init__(message)


class DeviceNotFoundError(GPUVideoToolsError):
    """Raised when a specified device cannot be found."""
    
    def __init__(self, device_spec: str, available_devices: list = None):
        self.device_spec = device_spec
        self.available_devices = available_devices or []
        message = f"Device '{device_spec}' not found."
        if available_devices:
            message += f"\n\nAvailable devices:\n"
            for dev in available_devices:
                message += f"  - {dev}\n"
        super().__init__(message)


class EncoderNotAvailableError(GPUVideoToolsError):
    """Raised when a requested encoder is not available for the chosen device."""
    
    def __init__(self, encoder: str, device: str, available_encoders: list = None):
        self.encoder = encoder
        self.device = device
        self.available_encoders = available_encoders or []
        message = f"Encoder '{encoder}' not available for device '{device}'."
        if available_encoders:
            message += f"\n\nAvailable encoders for this device:\n"
            for enc in available_encoders:
                message += f"  - {enc}\n"
        super().__init__(message)
