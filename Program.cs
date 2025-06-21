using System.IO;
using OpenCvSharp;
using FaceAiSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;
using SixLabors.ImageSharp.Processing;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

// Data models for JSON serialization
public class FaceEmbedding
{
    public string Name { get; set; } = "";
    public float[] Embeddings { get; set; } = Array.Empty<float>();
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public Dictionary<string, object> Metadata { get; set; } = new();
}

public class FaceDatabase
{
    public List<FaceEmbedding> Faces { get; set; } = new();
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
    public string Version { get; set; } = "1.0";
    public Dictionary<string, object> Settings { get; set; } = new();
}

class Program
{
    private static IFaceDetector? _detector;
    private static IFaceEmbeddingsGenerator? _recognizer;
    private static List<(string name, float[] embeddings)> _knownFaces = new();
    
    // JSON storage settings
    private static readonly string JSON_FILE_PATH = "face_database.json";
    private static readonly JsonSerializerOptions JSON_OPTIONS = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };
    
    static async Task Main(string[] args)
    {
        Console.WriteLine("Face Recognition Demo - With JSON Storage");

        try
        {
            InitializeFaceAI();
            LoadFaceDatabaseFromJSON();
            await RunWebcamDemo();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine("Press any key to exit");
            Console.ReadKey();
        }
    }

    static void InitializeFaceAI()
    {
        Console.WriteLine("Initializing FaceAiSharp models...");
        _detector = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks();
        _recognizer = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator();
        Console.WriteLine("Initialization Complete!");
    }

    static void LoadFaceDatabaseFromJSON()
    {
        try
        {
            if (!File.Exists(JSON_FILE_PATH))
            {
                Console.WriteLine("No existing face database found. Starting fresh.");
                SaveFaceDatabaseToJSON(); // Create empty database file
                return;
            }

            Console.WriteLine("Loading face database from JSON...");
            var jsonContent = File.ReadAllText(JSON_FILE_PATH);
            var database = JsonSerializer.Deserialize<FaceDatabase>(jsonContent, JSON_OPTIONS);

            if (database?.Faces != null)
            {
                _knownFaces.Clear();
                foreach (var face in database.Faces)
                {
                    _knownFaces.Add((face.Name, face.Embeddings));
                }

                Console.WriteLine($"Loaded {_knownFaces.Count} face embeddings from database");
                Console.WriteLine($"Database last updated: {database.LastUpdated:yyyy-MM-dd HH:mm:ss} UTC");
                Console.WriteLine($"Database version: {database.Version}");
                
                // Show loaded faces
                if (_knownFaces.Any())
                {
                    var uniqueNames = _knownFaces.GroupBy(f => f.name).ToList();
                    Console.WriteLine($"👥 Loaded people: {string.Join(", ", uniqueNames.Select(g => $"{g.Key} ({g.Count()})"))}");
                }
            }
            else
            {
                Console.WriteLine("Database file exists but appears to be empty or corrupted");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading face database: {ex.Message}");
            Console.WriteLine("Starting with empty database");
        }
    }

    static void SaveFaceDatabaseToJSON()
    {
        try
        {
            var database = new FaceDatabase
            {
                LastUpdated = DateTime.UtcNow,
                Version = "1.0"
            };

            // Add current threshold and other settings
            database.Settings["recognitionThreshold"] = 0.55f;
            database.Settings["embeddingDimension"] = _knownFaces.FirstOrDefault().embeddings?.Length ?? 512;

            // Convert known faces to FaceEmbedding objects
            foreach (var (name, embeddings) in _knownFaces)
            {
                var faceEmbedding = new FaceEmbedding
                {
                    Name = name,
                    Embeddings = embeddings,
                    CreatedAt = DateTime.UtcNow,
                    Id = Guid.NewGuid().ToString()
                };

                // Add metadata
                faceEmbedding.Metadata["embeddingMagnitude"] = Math.Sqrt(embeddings.Sum(x => x * x));
                faceEmbedding.Metadata["embeddingMean"] = embeddings.Average();

                database.Faces.Add(faceEmbedding);
            }

            var jsonContent = JsonSerializer.Serialize(database, JSON_OPTIONS);
            File.WriteAllText(JSON_FILE_PATH, jsonContent);

            Console.WriteLine($"Saved {database.Faces.Count} face embeddings to {JSON_FILE_PATH}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error saving face database: {ex.Message}");
        }
    }

    static void BackupFaceDatabase()
    {
        try
        {
            if (File.Exists(JSON_FILE_PATH))
            {
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var backupPath = $"face_database_backup_{timestamp}.json";
                File.Copy(JSON_FILE_PATH, backupPath);
                Console.WriteLine($"🔄 Database backed up to: {backupPath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Backup failed: {ex.Message}");
        }
    }

    static async Task RunWebcamDemo()
    {
        using var capture = new VideoCapture(0);
        
        if (!capture.IsOpened())
        {
            Console.WriteLine("Cannot open camera!");
            return;
        }

        Console.WriteLine("Camera opened successfully!");
        Console.WriteLine("Instructions:");
        Console.WriteLine("- Press 'E' to enroll largest face");
        Console.WriteLine("- Press 'A' to enroll all detected faces");
        Console.WriteLine("- Press 'R' to start recognition mode");
        Console.WriteLine("- Press 'D' to run face recognition diagnostics");
        Console.WriteLine("- Press 'L' to list all enrolled faces");
        Console.WriteLine("- Press 'C' to clear all enrolled faces");
        Console.WriteLine("- Press 'S' to save database manually");
        Console.WriteLine("- Press 'B' to backup database");
        Console.WriteLine("- Press 'I' to import faces from backup");
        Console.WriteLine("- Press 'Q' to quit");

        using var window = new OpenCvSharp.Window("Face Recognition with JSON Storage");
        var frame = new Mat();
        
        bool enrollMode = false;
        bool recognitionMode = false;
        bool enrollAllMode = false;

        while (true)
        {
            capture.Read(frame);
            if (frame.Empty()) continue;

            var imageBytes = frame.ToBytes(".jpg");
            using (var memoryStream = new MemoryStream(imageBytes))
            using (var image = SixLabors.ImageSharp.Image.Load<Rgb24>(memoryStream))
            {
                ProcessMultipleFaces(frame, image, enrollMode, recognitionMode, enrollAllMode);
            }
            
            window.ShowImage(frame);

            var key = Cv2.WaitKey(1) & 0xFF;
            switch (key)
            {
                case (int)'e':
                case (int)'E':
                    enrollMode = true;
                    recognitionMode = false;
                    enrollAllMode = false;
                    Console.WriteLine("\nEnroll mode: Press SPACE to capture largest face");
                    break;
                
                case (int)'a':
                case (int)'A':
                    enrollAllMode = true;
                    enrollMode = false;
                    recognitionMode = false;
                    Console.WriteLine("Enroll ALL mode: Press SPACE to capture all visible faces");
                    Console.WriteLine("Make sure all people are visible and positioned properly!");
                    Console.WriteLine("Face numbers will be FROZEN during enrollment process");
                    break;
                
                case (int)'r':
                case (int)'R':
                    enrollMode = false;
                    recognitionMode = true;
                    enrollAllMode = false;
                    Console.WriteLine("\nRecognition mode activated");
                    break;
                
                case (int)' ': // Spacebar
                    if (enrollMode || enrollAllMode)
                    {
                        Console.WriteLine("\nCAPTURING FRAME FOR ENROLLMENT...");
                        Console.WriteLine("Frame FROZEN - face positions are now locked!");
                        
                        // Capture the EXACT frame being displayed
                        var enrollBytes = frame.ToBytes(".jpg");
                        using (var enrollStream = new MemoryStream(enrollBytes))
                        using (var enrollImage = SixLabors.ImageSharp.Image.Load<Rgb24>(enrollStream))
                        {
                            if (enrollAllMode)
                            {
                                EnrollAllFacesImproved(enrollImage);
                            }
                            else
                            {
                                EnrollLargestFace(enrollImage);
                            }
                        }
                        
                        // Reset modes and auto-start recognition
                        enrollMode = false;
                        enrollAllMode = false;
                        
                        if (_knownFaces.Any())
                        {
                            recognitionMode = true;
                            Console.WriteLine("\nReturning to live camera feed...");
                            Console.WriteLine("Recognition mode activated automatically!");
                        }
                        else
                        {
                            Console.WriteLine("\nReturning to live camera feed...");
                        }
                    }
                    break;
                
                case (int)'d':
                case (int)'D':
                    Console.WriteLine("Running face recognition diagnostics...");
                    var diagBytes = frame.ToBytes(".jpg");
                    using (var diagStream = new MemoryStream(diagBytes))
                    using (var diagImage = SixLabors.ImageSharp.Image.Load<Rgb24>(diagStream))
                    {
                        DiagnoseFaceRecognition(diagImage);
                    }
                    break;
                
                case (int)'l':
                case (int)'L':
                    ListEnrolledFaces();
                    break;
                
                case (int)'c':
                case (int)'C':
                    ClearAllFaces();
                    break;

                case (int)'s':
                case (int)'S':
                    Console.WriteLine("Manually saving database...");
                    SaveFaceDatabaseToJSON();
                    break;

                case (int)'b':
                case (int)'B':
                    BackupFaceDatabase();
                    break;

                case (int)'i':
                case (int)'I':
                    ImportFacesFromBackup();
                    break;
                
                case (int)'q':
                case (int)'Q':
                    Console.WriteLine("Saving database before exit...");
                    SaveFaceDatabaseToJSON();
                    Console.WriteLine("Exiting...");
                    return;
            }
        }
    }

    static void ImportFacesFromBackup()
    {
        try
        {
            Console.WriteLine("\nIMPORT FACES FROM BACKUP");
            Console.WriteLine("===========================");

            // List available backup files
            var backupFiles = Directory.GetFiles(".", "face_database_backup_*.json")
                                      .OrderByDescending(f => File.GetLastWriteTime(f))
                                      .ToList();

            if (!backupFiles.Any())
            {
                Console.WriteLine("No backup files found");
                Console.WriteLine("Press any key to continue...");
                Console.ReadKey();
                return;
            }

            Console.WriteLine("Available backup files:");
            for (int i = 0; i < backupFiles.Count && i < 10; i++)
            {
                var info = new FileInfo(backupFiles[i]);
                Console.WriteLine($"{i + 1}. {info.Name} ({info.LastWriteTime:yyyy-MM-dd HH:mm})");
            }

            Console.Write("Enter backup number to import (or 0 to cancel): ");
            if (int.TryParse(Console.ReadLine(), out int choice) && choice > 0 && choice <= backupFiles.Count)
            {
                var selectedFile = backupFiles[choice - 1];
                
                Console.WriteLine($"Loading from {selectedFile}...");
                var jsonContent = File.ReadAllText(selectedFile);
                var database = JsonSerializer.Deserialize<FaceDatabase>(jsonContent, JSON_OPTIONS);

                if (database?.Faces != null)
                {
                    var beforeCount = _knownFaces.Count;
                    
                    // Option to merge or replace
                    Console.Write("Merge with current faces (M) or Replace all (R)? ");
                    var mergeChoice = Console.ReadKey().KeyChar.ToString().ToUpper();
                    Console.WriteLine();

                    if (mergeChoice == "R")
                    {
                        _knownFaces.Clear();
                    }

                    // Import faces
                    var importedCount = 0;
                    foreach (var face in database.Faces)
                    {
                        // Check for duplicates if merging
                        if (mergeChoice == "M" && _knownFaces.Any(f => f.name == face.Name))
                        {
                            Console.WriteLine($"Skipping duplicate: {face.Name}");
                            continue;
                        }

                        _knownFaces.Add((face.Name, face.Embeddings));
                        importedCount++;
                    }

                    Console.WriteLine($"Imported {importedCount} faces");
                    Console.WriteLine($"Total faces: {beforeCount} → {_knownFaces.Count}");
                    
                    // Auto-save after import
                    SaveFaceDatabaseToJSON();
                }
            }
            else
            {
                Console.WriteLine("Import cancelled");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Import failed: {ex.Message}");
        }

        Console.WriteLine("Press any key to continue...");
        Console.ReadKey();
    }

    static void ProcessMultipleFaces(Mat frame, SixLabors.ImageSharp.Image<Rgb24> image, bool enrollMode, bool recognitionMode, bool enrollAllMode)
    {
        try
        {
            var faces = _detector!.DetectFaces(image);
            
            int faceIndex = 0;
            foreach (var face in faces)
            {
                faceIndex++;
                
                var boundingBox = new Rect(
                    (int)face.Box.Left,
                    (int)face.Box.Top,
                    (int)face.Box.Width,
                    (int)face.Box.Height
                );

                var color = Scalar.Green;
                var label = $"Face {faceIndex}: Unknown";

                if (recognitionMode && _knownFaces.Any())
                {
                    try
                    {
                        using (var faceImage = ExtractFaceImage(image, face))
                        {
                            var matchEmbedding = _recognizer!.GenerateEmbedding(faceImage);
                            var match = FindBestMatch(matchEmbedding.ToArray());

                            if (match.HasValue)
                            {
                                label = $"{match.Value.name} ({match.Value.similarity:F2})";
                                color = Scalar.Blue;
                            }
                            else
                            {
                                label = $"Face {faceIndex}: Unknown";
                                color = Scalar.Red;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Recognition error for face {faceIndex}: {ex.Message}");
                        label = $"Face {faceIndex}: Error";
                        color = Scalar.Red;
                    }
                }
                else if (enrollMode)
                {
                    if (IsLargestFace(face, faces))
                    {
                        label = $"LARGEST - Press SPACE";
                        color = Scalar.Yellow;
                    }
                    else
                    {
                        label = $"Face {faceIndex}";
                        color = Scalar.Gray;
                    }
                }
                else if (enrollAllMode)
                {
                    label = $"Face {faceIndex} - Press SPACE";
                    color = Scalar.Yellow;
                }

                // Draw bounding box
                Cv2.Rectangle(frame, boundingBox, color, 2);
                
                // Draw label with background
                var labelSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.6, 2, out int baseline);
                var labelRect = new Rect(
                    boundingBox.X, 
                    boundingBox.Y - labelSize.Height - 15, 
                    Math.Max(labelSize.Width + 10, 120), 
                    labelSize.Height + baseline + 15
                );
                
                Cv2.Rectangle(frame, labelRect, Scalar.Black, -1);
                Cv2.PutText(frame, label, new OpenCvSharp.Point(boundingBox.X + 5, boundingBox.Y - 8),
                    HersheyFonts.HersheySimplex, 0.6, color, 2);

                // Draw large face number in the center of the box
                var numberLabel = faceIndex.ToString();
                var numberSize = Cv2.GetTextSize(numberLabel, HersheyFonts.HersheySimplex, 2.0, 3, out int numberBaseline);
                var numberX = boundingBox.X + (boundingBox.Width - numberSize.Width) / 2;
                var numberY = boundingBox.Y + (boundingBox.Height + numberSize.Height) / 2;
                
                Cv2.PutText(frame, numberLabel, new OpenCvSharp.Point(numberX, numberY),
                    HersheyFonts.HersheySimplex, 2.0, color, 3);
            }

            // Status display
            var status = enrollMode ? "ENROLL MODE (Largest Face)" : 
                        enrollAllMode ? "ENROLL ALL FACES MODE - Face numbers will be FIXED on capture" :
                        recognitionMode ? "RECOGNITION MODE" : 
                        "E=Enroll | A=Enroll All | R=Recognize | S=Save | B=Backup | I=Import | Q=Quit";

            // Status background
            var statusSize = Cv2.GetTextSize(status, HersheyFonts.HersheySimplex, 0.6, 2, out int statusBaseline);
            Cv2.Rectangle(frame, new Rect(5, 5, statusSize.Width + 10, statusSize.Height + statusBaseline + 10), 
                         Scalar.Black, -1);
            
            Cv2.PutText(frame, status, new OpenCvSharp.Point(10, 25),
                HersheyFonts.HersheySimplex, 0.6, Scalar.White, 2);

            // Face count and info  
            var info = $"Detected: {faces.Count()} | Known: {_knownFaces.Count} | DB: {(File.Exists(JSON_FILE_PATH) ? "GOOD" : "NOT GOOD")}";
            var infoSize = Cv2.GetTextSize(info, HersheyFonts.HersheySimplex, 0.5, 1, out int infoBaseline);
            Cv2.Rectangle(frame, new Rect(5, 35, infoSize.Width + 10, infoSize.Height + infoBaseline + 10), 
                         Scalar.Black, -1);
            
            Cv2.PutText(frame, info, new OpenCvSharp.Point(10, 50),
                HersheyFonts.HersheySimplex, 0.5, Scalar.Cyan, 1);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Processing Error: {ex.Message}");
        }
    }

    // IMPROVED: Better error handling and validation + auto-save
    static void EnrollAllFacesImproved(SixLabors.ImageSharp.Image<Rgb24> capturedImage)
    {
        try
        {
            Console.WriteLine("\nROCESSING CAPTURED FRAME...");
            
            // Get faces from the CAPTURED image (frozen frame)
            var faces = _detector!.DetectFaces(capturedImage).ToList();
            
            Console.WriteLine($"Debug: DetectFaces returned {faces.Count} faces");
            
            if (!faces.Any())
            {
                Console.WriteLine("\nNo faces detected in captured frame!");
                Console.WriteLine("Try getting closer to the camera or improving lighting");
                Console.WriteLine("Press any key to continue...");
                Console.ReadKey();
                return;
            }

            Console.WriteLine($"\nCAPTURED FRAME ANALYSIS:");
            Console.WriteLine($"Found {faces.Count} face(s) - positions are now FIXED");
            Console.WriteLine("==================================");

            // Show face mapping for user reference
            for (int i = 0; i < faces.Count; i++)
            {
                var face = faces[i];
                var position = GetFacePositionDescription(face.Box, capturedImage.Width, capturedImage.Height);
                var quality = IsFaceGoodQuality(face, capturedImage) ? "Good" : "Poor";
                Console.WriteLine($"  Face {i + 1}: {position} | Size: {face.Box.Width:F0}x{face.Box.Height:F0} | Quality: {quality}");
            }
            
            Console.WriteLine("===========================");
            Console.WriteLine("Now enrolling each face individually...\n");

            // Track enrollment progress
            var successfulEnrollments = 0;
            var beforeEnrollmentCount = _knownFaces.Count;

            // Process each face in the EXACT same order
            for (int i = 0; i < faces.Count; i++)
            {
                var face = faces[i];
                var faceNumber = i + 1;
                
                Console.WriteLine($"--- Enrolling Face {faceNumber}/{faces.Count} ---");
                Console.WriteLine($"Debug: Processing face at index {i} of {faces.Count}");
                
                var position = GetFacePositionDescription(face.Box, capturedImage.Width, capturedImage.Height);
                var quality = IsFaceGoodQuality(face, capturedImage);
                
                Console.WriteLine($"Position: {position}");
                Console.WriteLine($"Size: {face.Box.Width:F0} x {face.Box.Height:F0} pixels");
                Console.WriteLine($"Quality: {(quality ? "Good" : "Poor - may affect recognition")}");
                
                Console.Write($"👤 Enter name for Face {faceNumber} (or 'skip' to skip): ");
                
                // Force console to show the prompt immediately
                Console.Out.Flush();
                
                var name = Console.ReadLine()?.Trim();
                
                Console.WriteLine($"Debug: User entered '{name}' for face {faceNumber}");

                if (string.IsNullOrWhiteSpace(name))
                {
                    Console.WriteLine($"⏭Skipped Face {faceNumber} (empty name)");
                    Console.WriteLine();
                    continue; // This should continue to the next face
                }

                if (name.ToLower() == "skip")
                {
                    Console.WriteLine($"Skipped Face {faceNumber} by user choice");
                    Console.WriteLine();
                    continue; // This should continue to the next face
                }

                // Check if name already exists
                var existingNames = _knownFaces.Select(f => f.name).ToHashSet();
                if (existingNames.Contains(name))
                {
                    Console.Write($"Warning: Name '{name}' already exists. Continue? (y/n): ");
                    var confirmKey = Console.ReadKey(true);
                    Console.WriteLine(); // New line after key press
                    
                    if (confirmKey.KeyChar.ToString().ToLower() != "y")
                    {
                        Console.WriteLine($"Skipped Face {faceNumber}");
                        Console.WriteLine();
                        continue; // This should continue to the next face
                    }
                }

                try
                {
                    Console.WriteLine("Generating face embedding...");
                    
                    using (var faceImage = ExtractFaceImage(capturedImage, face))
                    {
                        var embedding = _recognizer!.GenerateEmbedding(faceImage);
                        var embeddingArray = embedding.ToArray();
                        
                        // Validate embedding quality
                        var magnitude = Math.Sqrt(embeddingArray.Sum(x => x * x));
                        if (Math.Abs(magnitude - 1.0) > 0.1) // Should be close to 1.0 for normalized embeddings
                        {
                            Console.WriteLine($"Warning: Embedding quality may be poor (magnitude: {magnitude:F3})");
                        }
                        
                        _knownFaces.Add((name, embeddingArray));
                        successfulEnrollments++;
                        
                        Console.WriteLine($"Face {faceNumber} enrolled as '{name}' successfully!");
                        Console.WriteLine($"Embedding magnitude: {magnitude:F3}");
                        Console.WriteLine($"Debug: Successfully added face {faceNumber}, continuing to next...");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to enroll Face {faceNumber}: {ex.Message}");
                    Console.WriteLine("This might be due to poor image quality or face detection issues");
                    Console.WriteLine($"Debug: Exception for face {faceNumber}, but continuing to next...");
                }
                
                Console.WriteLine(); // Add spacing between enrollments
                
                // Debug: Confirm we're continuing the loop
                if (i < faces.Count - 1)
                {
                    Console.WriteLine($"Debug: Moving to next face ({i + 2}/{faces.Count})...");
                }
            }

            // Final summary
            Console.WriteLine("🎉 ENROLLMENT SESSION COMPLETE!");
            Console.WriteLine("===============================");
            Console.WriteLine($"Successfully enrolled: {successfulEnrollments}/{faces.Count} faces");
            Console.WriteLine($"Total faces before: {beforeEnrollmentCount}");
            Console.WriteLine($"Total faces now: {_knownFaces.Count}");
            Console.WriteLine($"New faces added: {_knownFaces.Count - beforeEnrollmentCount}");
            
            // Auto-save to JSON after enrollment
            if (successfulEnrollments > 0)
            {
                Console.WriteLine("\nAuto-saving to database...");
                SaveFaceDatabaseToJSON();
                Console.WriteLine("Ready for recognition! The system will now automatically start recognizing enrolled faces.");
            }
            else
            {
                Console.WriteLine("\nNo faces were enrolled. Try again with better lighting or positioning.");
            }
            
            Console.WriteLine("\nPress any key to return to live feed...");
            Console.ReadKey();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Enrollment error: {ex.Message}");
            Console.WriteLine($"Debug: Exception in main enrollment method: {ex.StackTrace}");
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }

    static void ListEnrolledFaces()
    {
        Console.WriteLine("\nENROLLED FACES LIST");
        Console.WriteLine("==========================");
        
        if (!_knownFaces.Any())
        {
            Console.WriteLine("No faces enrolled yet");
            Console.WriteLine("Press 'A' to enroll multiple faces or 'E' to enroll single face");
        }
        else
        {
            var groupedFaces = _knownFaces.GroupBy(f => f.name).ToList();
            Console.WriteLine($"Total unique people: {groupedFaces.Count}");
            Console.WriteLine($"Total face embeddings: {_knownFaces.Count}");
            Console.WriteLine($"Database file: {(File.Exists(JSON_FILE_PATH) ? "Found" : "Missing")}");
            
            if (File.Exists(JSON_FILE_PATH))
            {
                var fileInfo = new FileInfo(JSON_FILE_PATH);
                Console.WriteLine($"-Last saved: {fileInfo.LastWriteTime:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine($"File size: {fileInfo.Length / 1024.0:F1} KB");
            }
            
            Console.WriteLine();
            
            int personIndex = 1;
            foreach (var group in groupedFaces)
            {
                Console.WriteLine($"{personIndex}. {group.Key} ({group.Count()} embedding{(group.Count() > 1 ? "s" : "")})");
                personIndex++;
            }
        }
        
        Console.WriteLine("\nPress any key to continue...");
        Console.ReadKey();
    }

    static void ClearAllFaces()
    {
        Console.WriteLine("\nCLEAR ALL ENROLLED FACES");
        Console.WriteLine($"Currently enrolled: {_knownFaces.Count} face embeddings");
        Console.Write("Are you sure you want to delete all enrolled faces? (type 'YES' to confirm): ");
        
        var confirmation = Console.ReadLine()?.Trim();
        if (confirmation == "YES")
        {
            if (_knownFaces.Any())
            {
                Console.WriteLine("Creating backup before clearing...");
                BackupFaceDatabase();
            }
            
            _knownFaces.Clear();
            SaveFaceDatabaseToJSON(); 
            Console.WriteLine("All enrolled faces cleared and database updated!");
        }
        else
        {
            Console.WriteLine("Operation cancelled");
        }
        
        Console.WriteLine("Press any key to continue...");
        Console.ReadKey();
    }

    static string GetFacePositionDescription(SixLabors.ImageSharp.RectangleF box, int imageWidth, int imageHeight)
    {
        var centerX = box.Left + box.Width / 2;
        var centerY = box.Top + box.Height / 2;
        
        string horizontal = centerX < imageWidth * 0.33 ? "Left" : 
                           centerX > imageWidth * 0.67 ? "Right" : "Center";
        string vertical = centerY < imageHeight * 0.33 ? "Top" : 
                         centerY > imageHeight * 0.67 ? "Bottom" : "Middle";
        
        // Add size descriptor
        var faceArea = box.Width * box.Height;
        var relativeSize = faceArea / (imageWidth * imageHeight);
        string sizeDesc = relativeSize > 0.15 ? "Large" : 
                         relativeSize > 0.05 ? "Medium" : "Small";
        
        return $"{sizeDesc} face, {horizontal}-{vertical}";
    }

    static SixLabors.ImageSharp.Image<Rgb24> ExtractFaceImage(SixLabors.ImageSharp.Image<Rgb24> originalImage, FaceAiSharp.FaceDetectorResult face)
    {
        var padding = 20;
        var left = Math.Max(0, (int)face.Box.Left - padding);
        var top = Math.Max(0, (int)face.Box.Top - padding);
        var right = Math.Min(originalImage.Width, (int)face.Box.Left + (int)face.Box.Width + padding);
        var bottom = Math.Min(originalImage.Height, (int)face.Box.Top + (int)face.Box.Height + padding);
        
        var width = right - left;
        var height = bottom - top;

        var faceRect = new Rectangle(left, top, width, height);
        var clonedImage = originalImage.Clone();
        clonedImage.Mutate(x => x.Crop(faceRect));
        
        return clonedImage;
    }

    static bool IsLargestFace(FaceAiSharp.FaceDetectorResult targetFace, IReadOnlyCollection<FaceAiSharp.FaceDetectorResult> allFaces)
    {
        var targetArea = targetFace.Box.Width * targetFace.Box.Height;
        return allFaces.All(face => face.Box.Width * face.Box.Height <= targetArea);
    }

    static void EnrollLargestFace(SixLabors.ImageSharp.Image<Rgb24> image)
    {
        try
        {
            var faces = _detector!.DetectFaces(image);
            if (!faces.Any())
            {
                Console.WriteLine("\nNo faces detected for enrollment!");
                return;
            }

            Console.Write($"\n👤 Enter name for the largest face: ");
            var name = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(name))
            {
                Console.WriteLine("Invalid name - enrollment cancelled");
                return;
            }

            var largestFace = faces.OrderByDescending(f => f.Box.Width * f.Box.Height).First();
            
            using (var faceImage = ExtractFaceImage(image, largestFace))
            {
                var embedding = _recognizer!.GenerateEmbedding(faceImage);
                _knownFaces.Add((name, embedding.ToArray()));
            }

            Console.WriteLine($"Largest face enrolled as '{name}' (Total: {_knownFaces.Count})");
            
            // Auto-save after single enrollment
            Console.WriteLine("Auto-saving to database...");
            SaveFaceDatabaseToJSON();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Enrollment error: {ex.Message}");
        }
    }

    static void DiagnoseFaceRecognition(SixLabors.ImageSharp.Image<Rgb24> image)
    {
        try
        {
            Console.WriteLine("\nFACE RECOGNITION DIAGNOSTICS");
            Console.WriteLine("==================================");
            
            var faces = _detector!.DetectFaces(image);
            Console.WriteLine($"Faces detected: {faces.Count()}");
            
            if (!faces.Any())
            {
                Console.WriteLine("No faces detected for diagnostics");
                return;
            }
            
            int faceIndex = 0;
            foreach (var face in faces)
            {
                faceIndex++;
                Console.WriteLine($"\n--- Face {faceIndex} Analysis ---");
                
                // Face quality metrics
                var faceArea = face.Box.Width * face.Box.Height;
                Console.WriteLine($"Face size: {face.Box.Width:F0}x{face.Box.Height:F0} (area: {faceArea:F0})");
                
                // Quality check
                var isGoodQuality = IsFaceGoodQuality(face, image);
                Console.WriteLine($"Face quality: {(isGoodQuality ? "Good" : "Poor")}");
                
                try
                {
                    using (var faceImage = ExtractFaceImage(image, face))
                    {
                        // Generate embedding
                        var currentEmbedding = _recognizer!.GenerateEmbedding(faceImage);
                        var embeddingArray = currentEmbedding.ToArray();
                        
                        // Analyze embedding quality
                        var magnitude = Math.Sqrt(embeddingArray.Sum(x => x * x));
                        var meanValue = embeddingArray.Average();
                        var stdDev = Math.Sqrt(embeddingArray.Average(x => Math.Pow(x - meanValue, 2)));
                        
                        Console.WriteLine($"Embedding stats:");
                        Console.WriteLine($"  Magnitude: {magnitude:F3} {(Math.Abs(magnitude - 1.0) < 0.01 ? "Normalized" : magnitude > 10.0 ? "GOOD" : magnitude > 5.0 ? "WEAK" : "NOT GOOD")}");
                        Console.WriteLine($"  Mean: {meanValue:F6}");
                        Console.WriteLine($"  Std Dev: {stdDev:F6} {(stdDev > 0.03 ? "Good" : "Poor")}");
                        Console.WriteLine($"  Note: FaceAiSharp uses normalized embeddings (magnitude ≈ 1.0)");
                        
                        if (_knownFaces.Any())
                        {
                            // Compare against ALL known faces with detailed similarity scores
                            Console.WriteLine($"\nSimilarity comparison against {_knownFaces.Count} known faces:");
                            
                            var personSimilarities = new Dictionary<string, List<float>>();
                            
                            foreach (var (name, knownEmbedding) in _knownFaces)
                            {
                                var similarity = CalculateCosineSimilarity(embeddingArray, knownEmbedding);
                                
                                if (!personSimilarities.ContainsKey(name))
                                    personSimilarities[name] = new List<float>();
                                
                                personSimilarities[name].Add(similarity);
                            }
                            
                            // Show best similarity for each person
                            foreach (var (name, similarities) in personSimilarities)
                            {
                                var maxSim = similarities.Max();
                                var avgSim = similarities.Average();
                                var count = similarities.Count;
                                
                                var status = maxSim > 0.55f ? "MATCH" : 
                                           maxSim > 0.45f ? "WEAK" : 
                                           maxSim > 0.35f ? "MAYBE" : "NO MATCH";
                                
                                Console.WriteLine($"  {name}: Max={maxSim:F3}, Avg={avgSim:F3}, Count={count} {status}");
                            }
                            
                            // Find best match using current algorithm
                            var bestMatch = FindBestMatch(embeddingArray);
                            if (bestMatch.HasValue)
                            {
                                Console.WriteLine($"algorithm result: {bestMatch.Value.name} ({bestMatch.Value.similarity:F3})");
                            }
                            else
                            {
                                Console.WriteLine($"algorithm result: NO MATCH (threshold: 0.55)");
                                
                                // Show what threshold would work
                                var allSimilarities = personSimilarities.SelectMany(p => p.Value).ToList();
                                if (allSimilarities.Any())
                                {
                                    var maxOverall = allSimilarities.Max();
                                    Console.WriteLine($"Highest similarity found: {maxOverall:F3}");
                                    Console.WriteLine($"Suggested threshold: {Math.Max(0.3f, maxOverall - 0.05f):F3}");
                                }
                            }
                        }
                        else
                        {
                            Console.WriteLine("No known faces enrolled for comparison");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to analyze face {faceIndex}: {ex.Message}");
                }
            }
            
            Console.WriteLine("\n" + "=============================");
            Console.WriteLine("Diagnostic Tips:");
            Console.WriteLine("- If similarity scores are 0.45-0.55, try lowering threshold");
            Console.WriteLine("- If embedding magnitude ≠ 1.0, check face extraction");
            Console.WriteLine("- If std dev < 0.03, face might be too blurry");
            Console.WriteLine("- Consider enrolling multiple angles for better recognition");
            Console.WriteLine("Database info:");
            if (File.Exists(JSON_FILE_PATH))
            {
                var fileInfo = new FileInfo(JSON_FILE_PATH);
                Console.WriteLine($"- Database file: {fileInfo.Length / 1024.0:F1} KB, modified {fileInfo.LastWriteTime:yyyy-MM-dd HH:mm}");
            }
            else
            {
                Console.WriteLine("- Database file: Not found");
            }
            Console.WriteLine("\nPress any key to continue...");
            Console.ReadKey();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Diagnostic failed: {ex.Message}");
        }
    }

    static bool IsFaceGoodQuality(FaceAiSharp.FaceDetectorResult face, SixLabors.ImageSharp.Image<Rgb24> image)
    {
        // Check face size (should be at least 80x80 pixels)
        if (face.Box.Width < 80 || face.Box.Height < 80)
            return false;
        
        // Check if face is too close to edges
        var margin = 20;
        if (face.Box.Left < margin || face.Box.Top < margin || 
            face.Box.Left + face.Box.Width > image.Width - margin ||
            face.Box.Top + face.Box.Height > image.Height - margin)
            return false;
        
        // Check aspect ratio (faces should be roughly rectangular)
        var aspectRatio = face.Box.Width / face.Box.Height;
        if (aspectRatio < 0.6 || aspectRatio > 1.6)
            return false;
        
        return true;
    }

    static float CalculateCosineSimilarity(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            return 0;

        // Since FaceAiSharp returns normalized embeddings (magnitude = 1.0),
        // we can use direct dot product for cosine similarity
        double dotProduct = 0;
        for (int i = 0; i < vector1.Length; i++)
        {
            dotProduct += vector1[i] * vector2[i];
        }

        // Convert from [-1, 1] range to [0, 1] range for easier interpretation
        return (float)Math.Max(0, dotProduct);
    }

    static (string name, float similarity)? FindBestMatch(float[] currentEmbedding)
    {
        float bestSimilarity = 0f;
        string? bestName = null;

        foreach (var (name, knownEmbedding) in _knownFaces)
        {
            var similarity = CalculateCosineSimilarity(currentEmbedding, knownEmbedding);

            if (similarity > bestSimilarity && similarity > 0.55f) // Lowered threshold for better recognition
            {
                bestSimilarity = similarity;
                bestName = name;
            }
        }

        return bestName != null ? (bestName, bestSimilarity) : null;
    }
}