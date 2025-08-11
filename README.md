# Multi-Camera YOLO Home Assistant MQTT

🎥 **Multi-camera YOLO object detection with Home Assistant integration via MQTT**

Monitor multiple RTSP camera streams, detect objects using YOLO AI models, and create individual Home Assistant sensors for each camera with seamless MQTT auto-discovery.

## ✨ Features

- 🔄 **Multi-Camera Support** - Monitor unlimited RTSP cameras
- 🤖 **YOLO Object Detection** - AI-powered object detection (people, pets, custom objects)  
- 🏠 **Home Assistant Integration** - Automatic MQTT discovery and individual sensor entities
- 🚀 **Raspberry Pi Optimized** - Runs efficiently on RPi 4 and other ARM devices
- ⚙️ **Easy Configuration** - Simple .env file configuration
- 📊 **Rich Logging** - Comprehensive logging and status reporting
- 🔧 **Flexible Deployment** - Cron job, continuous, or on-demand execution

## 🏗️ Project Structure

```
multi-cam-yolo-ha-mqtt/
├── main.py              # Main entry point
├── config.py            # Configuration and settings
├── monitor.py           # Main monitor orchestration
├── camera.py            # RTSP camera operations
├── detector.py          # YOLO object detection
├── mqtt_client.py       # MQTT/Home Assistant integration
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment configuration
├── .env                 # Your configuration (create from .env.example)
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/dennislwy/multi-cam-yolo-ha-mqtt.git
cd multi-cam-yolo-ha-mqtt

# Install dependencies
pip install -r requirements.txt

# Create configuration file
python main.py --create-env
cp .env.example .env
```

### 2. Configuration

Edit `.env` file with your settings:

```bash
# MQTT Configuration
MQTT_BROKER=192.168.0.88
MQTT_PORT=1883
MQTT_USERNAME=your_username
MQTT_PASSWORD=your_password

# Camera Configuration  
CAMERA_COUNT=2

# Camera 1
CAMERA_1_NAME=Front Door
CAMERA_1_RTSP_URL=rtsp://user:pass@192.168.0.5:554/stream1
CAMERA_1_LOCATION=Front Entrance
CAMERA_1_ENABLED=true

# Camera 2
CAMERA_2_NAME=Backyard
CAMERA_2_RTSP_URL=rtsp://user:pass@192.168.0.6:554/stream1
CAMERA_2_LOCATION=Backyard  
CAMERA_2_ENABLED=true
```

### 3. Testing

```bash
# Test system status
python main.py --status

# Validate camera connections
python main.py --validate

# Test object detection (all cameras)
python main.py --test

# Test specific camera
python main.py --test --camera "Front Door"
```

### 4. Production Deployment

```bash
# Setup Home Assistant discovery
python main.py --setup-ha

# Create systemd service for continuous monitoring
sudo nano /etc/systemd/system/camera-monitor.service
```

## 📋 Usage Examples

### Command Line Options

```bash
# Basic usage (single detection cycle)
python main.py

# Testing and validation
python main.py --test                    # Test all cameras
python main.py --test --camera "Front Door"  # Test specific camera
python main.py --validate               # Check camera connections
python main.py --status                 # Show system status

# Continuous monitoring (for debugging/manual)
python main.py --continuous

# Setup and configuration
python main.py --create-env             # Create example .env
python main.py --setup-ha               # Setup HA discovery only
```

### Systemd Service Setup (Recommended)

For continuous monitoring as a system service:

```bash
# Create service file
sudo nano /etc/systemd/system/camera-monitor.service
```

Add the following content:

```ini
[Unit]
Description=Multi-Camera YOLO Object Detection Monitor
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/multi-cam-yolo-ha-mqtt
ExecStart=/usr/bin/python3 /home/pi/multi-cam-yolo-ha-mqtt/main.py --continuous
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Environment variables (optional - can also use .env file)
Environment=PYTHONPATH=/home/pi/multi-cam-yolo-ha-mqtt

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable camera-monitor.service

# Start the service
sudo systemctl start camera-monitor.service

# Check service status
sudo systemctl status camera-monitor.service

# View logs
sudo journalctl -u camera-monitor.service -f
```

### Alternative: Cron Job Setup (Single Runs)

For periodic detection runs instead of continuous monitoring:

```bash
# Edit crontab
crontab -e

# Add this line for every minute detection
* * * * * cd /home/pi/multi-cam-yolo-ha-mqtt && /usr/bin/python3 main.py >> /var/log/camera_monitor_cron.log 2>&1

# Or every 5 minutes
*/5 * * * * cd /home/pi/multi-cam-yolo-ha-mqtt && /usr/bin/python3 main.py >> /var/log/camera_monitor_cron.log 2>&1
```

### Service Management Commands

```bash
# Service control
sudo systemctl start camera-monitor     # Start service
sudo systemctl stop camera-monitor      # Stop service
sudo systemctl restart camera-monitor   # Restart service
sudo systemctl status camera-monitor    # Check status

# Logs and monitoring
sudo journalctl -u camera-monitor -f    # Follow logs
sudo journalctl -u camera-monitor --since "1 hour ago"  # Recent logs
sudo journalctl -u camera-monitor --since today         # Today's logs

# Disable service
sudo systemctl disable camera-monitor   # Disable auto-start
sudo systemctl mask camera-monitor      # Completely disable
```

## ⚙️ Configuration Reference

### Core Settings

| Variable               | Default           | Description                                        |
| ---------------------- | ----------------- | -------------------------------------------------- |
| `MQTT_BROKER`          | `192.168.0.88`    | MQTT broker IP address                             |
| `MQTT_PORT`            | `1883`            | MQTT broker port                                   |
| `YOLO_MODEL_PATH`      | `yolov8n.pt`      | Path to YOLO model file                            |
| `CONFIDENCE_THRESHOLD` | `0.6`             | Detection confidence threshold (0.1-1.0)           |
| `INPUT_SIZE`           | `320`             | YOLO input image size (smaller = faster)           |
| `SUPPORTED_CLASSES`    | `person,dog,poop` | Comma-separated object classes                     |
| `CYCLE_DELAY`          | `60`              | Seconds between detection cycles (continuous mode) |

### Camera Configuration

For each camera (replace `X` with camera number 1, 2, 3...):

| Variable            | Required | Description                                |
| ------------------- | -------- | ------------------------------------------ |
| `CAMERA_X_NAME`     | ✅        | Human-readable camera name                 |
| `CAMERA_X_RTSP_URL` | ✅        | Full RTSP stream URL with credentials      |
| `CAMERA_X_LOCATION` | ❌        | Physical location (for HA area assignment) |
| `CAMERA_X_ENABLED`  | ❌        | Enable/disable camera (default: true)      |

## 🔧 Troubleshooting

### Common Issues

**"No cameras configured"**
- Check `CAMERA_COUNT` is set correctly
- Ensure `CAMERA_X_NAME` and `CAMERA_X_RTSP_URL` are defined for each camera
- Run `python main.py --status` to check configuration

**"Failed to capture frame"**
- Verify RTSP URL and credentials
- Test camera connection: `python main.py --validate`
- Check network connectivity to camera
- Increase `RTSP_TIMEOUT` if needed

**"YOLO model not found"**
- Ensure YOLOv8 model file exists: `ls yolov8n.pt`
- Download model: `yolo export model=yolov8n.pt format=pt`

**MQTT connection failed**
- Verify MQTT broker IP and credentials
- Check network connectivity: `ping {MQTT_BROKER}`
- Test MQTT connection: `python main.py --setup-ha`

### Performance Optimization

**Raspberry Pi 4 Optimization:**
- Use `INPUT_SIZE=320` for faster inference
- Set `CONFIDENCE_THRESHOLD=0.6` or higher
- Consider NCNN export for better ARM performance:
  ```bash
  yolo export model=yolov8n.pt format=ncnn
  ```

**Memory Issues:**
- Reduce `INPUT_SIZE` to 256 or lower
- Increase swap file size
- Monitor with `htop` during operation

## 📊 Monitoring and Logs

### Log Files
- Application logs: `/var/log/camera_monitor.log` (configurable)  
- System service logs: `journalctl -u camera-monitor`

### Status Monitoring
```bash
# Real-time log monitoring (service)
sudo journalctl -u camera-monitor.service -f

# Real-time log monitoring (file)
tail -f /var/log/camera_monitor.log

# System status
python main.py --status

# Camera validation  
python main.py --validate

# Service status
sudo systemctl status camera-monitor
```

## 🔄 Updates and Maintenance

### Updating YOLO Model
```bash
# Download newer model
yolo export model=yolov8s.pt format=pt

# Update .env
YOLO_MODEL_PATH=yolov8s.pt
```

### Adding New Cameras
1. Edit `.env` file
2. Increment `CAMERA_COUNT`
3. Add new `CAMERA_X_*` variables  
4. Test: `python main.py --validate`
5. Setup HA discovery: `python main.py --setup-ha`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection models
- [Home Assistant](https://www.home-assistant.io/) - Home automation platform
- [Paho MQTT](https://github.com/eclipse/paho.mqtt.python) - MQTT client library

## 📞 Support

- 📖 [Documentation](https://github.com/yourusername/multi-cam-yolo-ha-mqtt/wiki)
- 🐛 [Issue Tracker](https://github.com/yourusername/multi-cam-yolo-ha-mqtt/issues)
- 💬 [Discussions](https://github.com/yourusername/multi-cam-yolo-ha-mqtt/discussions)