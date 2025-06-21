# Face Recognition System - FaceAiSharp Edition

A professional real-time face detection and recognition system built with C#, FaceAiSharp, and OpenCV. Features persistent JSON storage, advanced diagnostics, and production-ready face embeddings.

## 🌟 Features

### Core Functionality

- **Real-time face detection** using state-of-the-art AI models
- **High-accuracy face recognition** with deep learning embeddings
- **Persistent JSON storage** for face database management
- **Multi-face enrollment** with intelligent positioning
- **Professional diagnostics** with detailed similarity analysis
- **Database backup and restore** capabilities

### Advanced AI Models

- **FaceAiSharp integration** - Production-grade face detection and recognition
- **Normalized embeddings** - High-quality 512-dimensional face vectors
- **Cosine similarity matching** - Industry-standard comparison algorithm
- **Quality assessment** - Automatic face quality validation

### Data Management

- **JSON database storage** - Human-readable and portable
- **Automatic backups** - Timestamped database snapshots
- **Import/Export functionality** - Easy data migration
- **Metadata tracking** - Embedding statistics and timestamps

## 📋 Prerequisites

### Required Dependencies

- **.NET 6.0 or later**
- **FaceAiSharp** NuGet package
- **OpenCvSharp4** NuGet package
- **SixLabors.ImageSharp** NuGet package
- **Webcam** or video capture device

### Hardware Requirements

- **Minimum**: 4GB RAM, dual-core processor, integrated webcam
- **Recommended**: 8GB+ RAM, quad-core processor, dedicated webcam
- **GPU**: Not required (CPU-based inference)

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd face-recognition-faceaisharp
   ```

2. **Install required packages**

   ```bash
   dotnet add package FaceAiSharp
   dotnet add package OpenCvSharp4
   dotnet add package OpenCvSharp4.runtime.win  # For Windows
   dotnet add package SixLabors.ImageSharp
   ```

3. **Build and run**

   ```bash
   dotnet build
   dotnet run
   ```

4. **First run initialization**
   - FaceAiSharp models will be automatically downloaded on first use
   - A JSON database file will be created automatically

## 🎮 Usage Guide

### Basic Controls

| Key   | Function                                      |
| ----- | --------------------------------------------- |
| **E** | Enroll largest detected face                  |
| **A** | Enroll all detected faces (multi-person mode) |
| **R** | Start recognition mode                        |
| **D** | Run detailed face recognition diagnostics     |
| **L** | List all enrolled faces with statistics       |
| **C** | Clear all enrolled faces (with confirmation)  |
| **S** | Manually save database                        |
| **B** | Create database backup                        |
| **I** | Import faces from backup                      |
| **Q** | Quit (auto-saves database)                    |

### Enrollment Workflow

#### Single Face Enrollment

1. Press **E** to enter enrollment mode
2. Position your face in view (largest face will be highlighted)
3. Press **SPACE** to capture
4. Enter your name when prompted
5. Recognition mode automatically activates

#### Multi-Face Enrollment

1. Press **A** to enter multi-face enrollment mode
2. Ensure all people are visible and well-positioned
3. Press **SPACE** to freeze the frame
4. Enter names for each detected face (or type "skip" to skip)
5. System automatically saves and starts recognition

### Recognition Mode

- Press **R** to start real-time recognition
- Known faces show name and confidence score
- Unknown faces are marked in red
- Confidence scores range from 0.0 to 1.0 (higher = more confident)

### Advanced Features

#### Database Management

- **Auto-save**: Database saves automatically after enrollment
- **Backup**: Press **B** to create timestamped backups
- **Import**: Press **I** to restore from backup files
- **Manual save**: Press **S** to force save current state

#### Diagnostics Mode

Press **D** for comprehensive face analysis:

- **Face quality assessment** - Size, position, clarity evaluation
- **Embedding statistics** - Magnitude, mean, standard deviation
- **Similarity matrix** - Detailed comparison against all known faces
- **Threshold recommendations** - Suggested confidence thresholds
- **Database health check** - File status and statistics

## 🔧 Technical Details

### Face Detection

- **Model**: FaceAiSharp's pre-trained detection models
- **Accuracy**: High precision with minimal false positives
- **Speed**: Real-time performance on standard hardware
- **Multi-face**: Supports multiple faces simultaneously

### Face Recognition

- **Embeddings**: 512-dimensional normalized vectors
- **Algorithm**: Deep learning-based feature extraction
- **Similarity**: Cosine similarity with 0.55 default threshold
- **Quality**: Automatic face quality validation

### Feature Extraction Process

1. **Face detection** - Locate faces in image
2. **Face extraction** - Crop and pad face regions
3. **Preprocessing** - Normalize and resize for model input
4. **Embedding generation** - Extract 512D feature vector
5. **Normalization** - Ensure unit magnitude for consistent comparison

### Similarity Calculation

```
Cosine Similarity = dot(embedding1, embedding2) / (||embedding1|| × ||embedding2||)
Since embeddings are normalized: Similarity = dot(embedding1, embedding2)
```

### Database Schema

```json
{
  "faces": [
    {
      "name": "John Doe",
      "embeddings": [0.1234, -0.5678, ...],
      "createdAt": "2025-06-21T10:30:00Z",
      "id": "uuid-string",
      "metadata": {
        "embeddingMagnitude": 1.0,
        "embeddingMean": 0.0
      }
    }
  ],
  "lastUpdated": "2025-06-21T10:30:00Z",
  "version": "1.0",
  "settings": {
    "recognitionThreshold": 0.55,
    "embeddingDimension": 512
  }
}
```

## ⚙️ Configuration

### Recognition Threshold

The default threshold of 0.55 works well for most scenarios:

- **Higher values (0.6-0.8)**: More strict, fewer false positives
- **Lower values (0.4-0.55)**: More permissive, may increase false positives
- **Diagnostic mode** suggests optimal thresholds based on your data

### Face Quality Requirements

- **Minimum size**: 80x80 pixels
- **Aspect ratio**: 0.6 - 1.6 (roughly rectangular)
- **Position**: Not too close to image edges
- **Clarity**: Sufficient detail for feature extraction

## 🛠️ Troubleshooting

### Common Issues

**"Cannot open camera"**

- Ensure webcam is connected and available
- Close other applications using the camera
- Try different camera indices if multiple cameras present

**"Poor recognition accuracy"**

- Run diagnostics mode (**D**) to analyze face quality
- Ensure good lighting conditions
- Re-enroll faces with better positioning
- Check similarity scores in diagnostic output

**"No faces detected"**

- Improve lighting conditions
- Move closer to camera
- Ensure face is clearly visible and upright
- Check camera focus and clarity

**"Low similarity scores"**

- Face may be at different angle than enrollment
- Lighting conditions may differ significantly
- Consider enrolling multiple images of the same person
- Check diagnostic recommendations for threshold adjustment

### Performance Optimization

- **Memory**: Close other applications to free RAM
- **CPU**: Reduce camera resolution if performance is poor
- **Storage**: Clean up old backup files periodically
- **Database**: Keep face database under 100 people for optimal speed

### Database Issues

- **Corrupted database**: Use backup files (**I** key) to restore
- **Missing database**: System creates new database automatically
- **Large file size**: Consider removing unused faces (**C** key)

## 📊 Quality Metrics

### Embedding Quality Indicators

- **Magnitude**: Should be ≈ 1.0 (normalized embeddings)
- **Standard deviation**: > 0.03 indicates good feature diversity
- **Mean**: Should be close to 0.0 for balanced features

### Recognition Performance

- **Good match**: Similarity > 0.65
- **Acceptable match**: Similarity 0.55 - 0.65
- **Uncertain**: Similarity 0.45 - 0.55 (consider re-enrollment)
- **No match**: Similarity < 0.45

## 📁 File Structure

```
project/
├── Program.cs                    # Main application code
├── face_database.json          # Primary face database
├── face_database_backup_*.json # Timestamped backups
└── bin/                        # Compiled application
```

## 🔒 Privacy and Security

- **Local storage**: All data stays on your machine
- **No cloud connectivity**: Completely offline operation
- **Encrypted storage**: Consider encrypting the database file for sensitive deployments
- **Access control**: Implement file permissions as needed

## 🚨 Production Considerations

- **Backup strategy**: Regular automated backups recommended
- **Monitoring**: Implement logging for production deployments
- **Scaling**: Consider database size limits (< 1000 faces recommended)
- **Hardware**: Ensure adequate CPU and memory for target user count

## 📈 Future Enhancements

- Database encryption support
- Network-based face database sharing
- REST API for integration
- Batch processing capabilities
- Advanced anti-spoofing measures

## 📄 License

MIT License

## 🙏 Acknowledgments

- **FaceAiSharp** - High-quality face recognition models
- **OpenCV** - Computer vision foundation
- **SixLabors.ImageSharp** - Modern image processing
- **Microsoft** - .NET ecosystem
