import torch
import cvzone
import cv2
from torchvision import transforms
import torchvision

# Memuat model Faster R-CNN dengan ResNet50-FPN yang sudah dilatih sebelumnya
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval() # Mengatur model ke mode evaluasi (penting untuk inferensi)

# Memuat nama-nama kelas dari file 'classes.txt'
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Inisialisasi webcam
cap = cv2.VideoCapture(0) # 0 adalah ID default untuk webcam internal
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    exit()

# Transformasi gambar untuk model PyTorch
image_transform = transforms.ToTensor()

while True:
    ret, frame = cap.read() # Membaca frame dari webcam
    if not ret:
        print("Error: Tidak dapat membaca frame dari webcam.")
        break

    # Mengubah ukuran frame agar konsisten (opsional, tapi disarankan)
    # Sesuaikan resolusi sesuai kebutuhan Anda
    frame = cv2.resize(frame, (640, 480))

    # Mengubah frame dari format BGR (OpenCV) ke RGB (PyTorch)
    # dan kemudian mengonversinya menjadi tensor PyTorch
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = image_transform(img_rgb)

    with torch.no_grad(): # Nonaktifkan perhitungan gradien untuk inferensi
        pred = model([img_tensor]) # Melakukan inferensi pada frame
        
        # Mengambil kotak pembatas (bbox), skor, dan label dari prediksi
        bbox, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']
        
        # Menentukan ambang kepercayaan (confidence threshold)
        # Objek dengan skor di bawah ambang ini akan diabaikan
        confidence_threshold = 0.70 
        
        # Memfilter deteksi berdasarkan ambang kepercayaan
        # argwhere mengembalikan indeks elemen yang memenuhi kondisi
        high_conf_indices = torch.argwhere(scores > confidence_threshold).squeeze()
        
        # Pastikan high_conf_indices adalah tensor 1D atau 0D (skalar)
        if high_conf_indices.dim() == 0: # Jika hanya ada satu deteksi
            high_conf_indices = high_conf_indices.unsqueeze(0) # Ubah menjadi tensor 1D

        # Iterasi melalui deteksi yang memiliki kepercayaan tinggi
        for i in high_conf_indices:
            x, y, x2, y2 = bbox[i].numpy().astype('int') # Koordinat kotak pembatas (x1, y1, x2, y2)
            classname_id = labels[i].numpy().astype('int') # ID kelas
            
            # Pastikan classname_id valid sebelum mengakses classnames
            if 0 < classname_id <= len(classnames):
                class_detected = classnames[classname_id - 1] # Mengambil nama kelas (perhatikan indeks 0 jika kelas dimulai dari 1)
                
                # Menggambar kotak pembatas pada frame
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2) # Warna hijau, ketebalan 2
                
                # Menambahkan teks nama kelas dan skor kepercayaan
                score_text = f"{class_detected}: {scores[i]:.2f}"
                cvzone.putTextRect(frame, score_text, [x, y - 10], scale=1.5, thickness=2, offset=5, border=2)
            else:
                print(f"Peringatan: ID kelas tidak valid terdeteksi: {classname_id}")

    # Menampilkan frame dengan deteksi
    cv2.imshow('Realtime Object Detection', frame)

    # Menekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan objek VideoCapture dan menutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()