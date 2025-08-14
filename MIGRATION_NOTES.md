# Migration from CameraHandler to RTSPVideoStream

## Overview

This document outlines the migration from the deprecated `CameraHandler` class to the new `RTSPVideoStream` class for improved performance, thread safety, and additional features.

## Changes Made

### 1. Deprecated CameraHandler
- Added deprecation warnings to `camera.py`
- Updated `__init__.py` to handle deprecation gracefully
- CameraHandler will be removed in a future version

### 2. Updated Monitor Class (`monitor.py`)
- **Removed**: `CameraHandler` dependency
- **Added**: `RTSPVideoStream` management with `camera_streams` dictionary
- **Added**: `_initialize_camera_streams()` method to set up streams for all cameras
- **Added**: `cleanup_camera_streams()` method for proper resource cleanup
- **Updated**: Frame capture logic in `run_camera_detection_cycle()` to use stream.read()
- **Updated**: `cleanup()` method to use new stream cleanup

### 3. Updated MultiCameraProcessor (`multi_camera_processor.py`)
- **Removed**: `CameraHandler` dependency
- **Updated**: `process_camera_worker()` to get frames from monitor's camera streams
- **Updated**: `validate_all_cameras()` to use stream health checking instead of connection validation
- **Updated**: `cleanup()` method (no longer needs camera handler cleanup)

## Key Improvements

### Performance Benefits
- **Thread-safe operations**: Multiple threads can safely read frames
- **Reduced latency**: Single frame buffering for real-time streaming
- **Better resource management**: Automatic connection management per camera

### New Features Available
- **Video recording**: Built-in recording capabilities with multiple codecs
- **Snapshot capture**: Timestamp-based image capture
- **Stream health monitoring**: Detailed status information
- **Automatic reconnection**: Configurable retry logic with circuit breaker pattern

### Better Error Handling
- **Stream status monitoring**: `is_running()`, `get_status()` methods
- **Graceful failure handling**: Proper cleanup on connection failures
- **Circuit breaker integration**: Seamless integration with existing failure tracking

## Breaking Changes

### For End Users
- **None**: The public API remains the same
- Deprecation warnings will be shown when using old imports

### For Developers
- `CameraHandler.capture_frame_from_rtsp()` → `RTSPVideoStream.read()`
- Manual connection management → Automatic stream lifecycle management
- Connection caching → Per-camera stream instances

## Migration Benefits

1. **Better Resource Management**
   - Automatic cleanup of video streams
   - Proper thread lifecycle management
   - Memory leak prevention

2. **Enhanced Monitoring**
   - Real-time stream health status
   - Connection stability metrics
   - Performance monitoring capabilities

3. **Future-Ready Architecture**
   - Support for video recording
   - Snapshot capabilities
   - Extensible for additional features

## Configuration Changes

### Settings Integration
- Uses existing `rtsp_timeout` setting for reconnection delays
- Leverages existing circuit breaker settings
- No new configuration required

### Stream Management
- Each camera gets its own `RTSPVideoStream` instance
- Streams are initialized during monitor startup
- Automatic cleanup during shutdown

## Testing Recommendations

1. **Verify Stream Initialization**
   ```python
   # Check that all camera streams are created and running
   assert all(stream.is_running() for stream in monitor.camera_streams.values())
   ```

2. **Test Frame Capture**
   ```python
   # Ensure frames are being captured from all cameras
   for camera in cameras:
       frame = monitor.camera_streams[camera["id"]].read()
       assert frame is not None
   ```

3. **Validate Cleanup**
   ```python
   # Ensure proper cleanup
   monitor.cleanup_camera_streams()
   assert len(monitor.camera_streams) == 0
   ```

## Timeline

- **Current**: CameraHandler marked as deprecated with warnings
- **Next Release**: CameraHandler will be removed entirely
- **Migration Period**: Both systems work in parallel with warnings

## Support

If you encounter issues during migration:
1. Check stream status with `stream.get_status()`
2. Verify camera URLs are accessible
3. Review logs for connection issues
4. Ensure proper cleanup in error handling
