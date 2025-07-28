import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Veri Seti Sınıfı
class ChangeDetectionDataset(Dataset):
    def __init__(self, pre_images, post_images, masks, transform=None):
        self.pre_images = pre_images
        self.post_images = post_images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.pre_images)
    
    def __getitem__(self, idx):
        pre_img = self.pre_images[idx]
        post_img = self.post_images[idx]
        mask = self.masks[idx]
        
        # Maskeyi [H,W,3] -> [H,W,1] ve [0,1] aralığına getir
        if mask.ndim == 2:
            mask = mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1)  # [H,W,1]
        else:
            mask = mask[:,:,0:1].astype(np.float32) / 255.0
        
        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)
            # Maskeye sadece ToTensor uygula (normalize etme)
            mask = transforms.ToTensor()(mask)
        
        return pre_img, post_img, mask

# UNet Bileşenleri
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Değişim Tespiti UNet Modeli
class ChangeDetectionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Pre-image encoder
        self.pre_inc = DoubleConv(n_channels, 64)
        self.pre_down1 = Down(64, 128)
        self.pre_down2 = Down(128, 256)
        self.pre_down3 = Down(256, 512)
        self.pre_down4 = Down(512, 1024)
        
        # Post-image encoder
        self.post_inc = DoubleConv(n_channels, 64)
        self.post_down1 = Down(64, 128)
        self.post_down2 = Down(128, 256)
        self.post_down3 = Down(256, 512)
        self.post_down4 = Down(512, 1024)
        
        # Feature difference and decoder
        self.up1 = Up(2048, 512, bilinear)
        self.up2 = Up(1024, 256, bilinear)
        self.up3 = Up(512, 128, bilinear)
        self.up4 = Up(256, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, pre_x, post_x):
        # Encode pre-image
        pre_x1 = self.pre_inc(pre_x)
        pre_x2 = self.pre_down1(pre_x1)
        pre_x3 = self.pre_down2(pre_x2)
        pre_x4 = self.pre_down3(pre_x3)
        pre_x5 = self.pre_down4(pre_x4)
        
        # Encode post-image
        post_x1 = self.post_inc(post_x)
        post_x2 = self.post_down1(post_x1)
        post_x3 = self.post_down2(post_x2)
        post_x4 = self.post_down3(post_x3)
        post_x5 = self.post_down4(post_x4)
        
        # Feature difference and decoding
        x = torch.cat([pre_x5, post_x5], dim=1)
        x = self.up1(x, torch.cat([pre_x4, post_x4], dim=1))
        x = self.up2(x, torch.cat([pre_x3, post_x3], dim=1))
        x = self.up3(x, torch.cat([pre_x2, post_x2], dim=1))
        x = self.up4(x, torch.cat([pre_x1, post_x1], dim=1))
        logits = self.outc(x)
        return torch.sigmoid(logits)

# Ana Değişim Tespit Sınıfı
class ChangeDetector:
    def __init__(self):
        self.image1 = None
        self.image2 = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_images(self, path1, path2):
        if path1:
            self.image1 = self._load_image(path1)
        if path2:
            self.image2 = self._load_image(path2)
        return self.image1 is not None and self.image2 is not None
    
    def _load_image(self, path):
        if not os.path.exists(path):
            print(f"Dosya bulunamadı: {path}")
            return None
        
        try:
            if path.lower().endswith(('.tif', '.tiff', '.geotiff')):
                import rasterio
                with rasterio.open(path) as src:
                    image = src.read()
                    image = np.transpose(image, (1, 2, 0)) if image.shape[0] == 3 else image[0]
                    image = image.astype(np.uint8)
            else:
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        except Exception as e:
            print(f"Yükleme hatası: {e}")
            return None
    
    def load_model(self, model_path):
        self.model = ChangeDetectionUNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def train_model(self, pre_images, post_images, masks, epochs=10, batch_size=2):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = ChangeDetectionDataset(pre_images, post_images, masks, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = ChangeDetectionUNet().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for pre, post, mask in loader:
                pre = pre.to(self.device)
                post = post.to(self.device)
                mask = mask.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(pre, post)
                loss = criterion(outputs, mask)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}')
        
        return self.model
    
    def detect_changes(self, method='ssim', threshold=0.5):
        if method == 'ssim':
            return self._detect_with_ssim(threshold)
        elif method == 'diff':
            return self._detect_with_diff(threshold)
        elif method == 'unet' and self.model is not None:
            return self._detect_with_unet(threshold)
        else:
            print("Geçersiz yöntem veya model yüklenmedi!")
            return None
    
    def _detect_with_ssim(self, threshold):
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_RGB2GRAY)
        
        score, diff = ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return thresh
    
    def _detect_with_diff(self, threshold):
        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        threshold_value = int(threshold * 255)
        _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return cleaned
    
    def _detect_with_unet(self, threshold):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pre_img = transform(self.image1).unsqueeze(0).to(self.device)
        post_img = transform(self.image2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(pre_img, post_img)
        
        change_map = (output.squeeze().cpu().numpy() > threshold).astype('uint8') * 255
        return change_map

# GUI Uygulaması
class ChangeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uydu Görüntüsü Değişim Analizi")
        self.detector = ChangeDetector()
        self.method_var = tk.StringVar(value="ssim")
        self.create_widgets()
    
    def create_widgets(self):
        # Görüntü yükleme butonları
        tk.Button(self.root, text="Önceki Görüntüyü Yükle", command=self.load_image1).pack(pady=5)
        tk.Button(self.root, text="Sonraki Görüntüyü Yükle", command=self.load_image2).pack(pady=5)
        
        # Yöntem seçimi
        tk.Label(self.root, text="Yöntem Seçin:").pack()
        tk.Radiobutton(self.root, text="SSIM", variable=self.method_var, value="ssim").pack()
        tk.Radiobutton(self.root, text="Fark Alma", variable=self.method_var, value="diff").pack()
        tk.Radiobutton(self.root, text="U-Net (Derin Öğrenme)", variable=self.method_var, value="unet").pack()
        
        # Eşik değeri
        tk.Label(self.root, text="Eşik Değeri (0-1):").pack()
        self.threshold_slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack()
        
        # Eğitim parametreleri
        tk.Label(self.root, text="Epoch Sayısı:").pack()
        self.epochs_entry = tk.Entry(self.root)
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.pack()
        
        tk.Label(self.root, text="Batch Boyutu:").pack()
        self.batch_entry = tk.Entry(self.root)
        self.batch_entry.insert(0, "2")
        self.batch_entry.pack()
        
        # Butonlar
        tk.Button(self.root, text="Değişiklikleri Tespit Et", command=self.detect_changes).pack(pady=10)
        tk.Button(self.root, text="Model Yükle", command=self.load_model).pack(pady=5)
        tk.Button(self.root, text="Modeli Eğit", command=self.train_model_dialog).pack(pady=5)
        
        # Görüntü gösterme alanları
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()
        
        self.before_label = tk.Label(self.image_frame)
        self.before_label.grid(row=0, column=0, padx=5)
        
        self.after_label = tk.Label(self.image_frame)
        self.after_label.grid(row=0, column=1, padx=5)
        
        self.change_label = tk.Label(self.image_frame)
        self.change_label.grid(row=0, column=2, padx=5)
    
    def load_image1(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.tif *.tiff *.geotiff *.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ])
        if path:
            if self.detector.load_images(path, ""):
                self.show_image(self.detector.image1, self.before_label)
    
    def load_image2(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.tif *.tiff *.geotiff *.png *.jpg *.jpeg"),
            ("All files", "*.*")
        ])
        if path:
            if self.detector.load_images("", path):
                self.show_image(self.detector.image2, self.after_label)
    
    def show_image(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(image)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo
    
    def detect_changes(self):
        if self.detector.image1 is None or self.detector.image2 is None:
            messagebox.showerror("Hata", "Lütfen her iki görüntüyü de yükleyin!")
            return
        
        method = self.method_var.get()
        threshold = self.threshold_slider.get()
        change_map = self.detector.detect_changes(method, threshold)
        
        if change_map is not None:
            change_map_rgb = cv2.cvtColor(change_map, cv2.COLOR_GRAY2RGB)
            self.show_image(change_map_rgb, self.change_label)
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(self.detector.image1)
            plt.title("Önce")
            plt.subplot(1, 3, 2)
            plt.imshow(self.detector.image2)
            plt.title("Sonra")
            plt.subplot(1, 3, 3)
            plt.imshow(change_map, cmap='gray')
            plt.title("Değişim Haritası")
            plt.show()
    
    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            self.detector.load_model(path)
            messagebox.showinfo("Bilgi", "Model başarıyla yüklendi!")
    
    def train_model_dialog(self):
        messagebox.showinfo("Bilgi", "Önce tüm öncesi görüntüleri seçin")
        pre_files = filedialog.askopenfilenames(filetypes=[("Image files", "*.tif *.png *.jpg")])
        
        messagebox.showinfo("Bilgi", "Şimdi tüm sonrası görüntüleri seçin (aynı sırada)")
        post_files = filedialog.askopenfilenames(filetypes=[("Image files", "*.tif *.png *.jpg")])
        
        messagebox.showinfo("Bilgi", "Şimdi tüm maskeleri seçin (binary, siyah-beyaz)")
        mask_files = filedialog.askopenfilenames(filetypes=[("Image files", "*.tif *.png *.jpg")])
        
        if len(pre_files) != len(post_files) or len(pre_files) != len(mask_files):
            messagebox.showerror("Hata", "Aynı sayıda dosya seçmelisiniz!")
            return
        
        try:
            # Eğitim başlıyor uyarısı
            messagebox.showinfo("Bilgi", "Model şu an eğitiliyor! Lütfen bekleyin...")
            # Görüntüleri yükle
            pre_images = [self.load_image_as_array(f) for f in pre_files]
            post_images = [self.load_image_as_array(f) for f in post_files]
            
            # Maskeleri yükle (tek kanal olarak)
            masks = []
            for f in mask_files:
                m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise ValueError(f"Maske yüklenemedi: {f}")
                masks.append(m)
            
            # Eğitim parametreleri
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_entry.get())
            
            # Eğitimi başlat
            self.detector.train_model(pre_images, post_images, masks, epochs, batch_size)
            # Eğitim tamamlandı uyarısı
            messagebox.showinfo("Bilgi", "Model eğitimi tamamlandı!")
            # Modeli kaydet
            save_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
            if save_path:
                torch.save(self.detector.model.state_dict(), save_path)
                messagebox.showinfo("Başarılı", f"Model başarıyla kaydedildi: {save_path}")
                
        except Exception as e:
            # Hata olursa pencereyi kapatma, sadece mesaj göster ve terminale yaz
            messagebox.showerror("Hata", f"Eğitim sırasında hata oluştu: {str(e)}\nDetaylar için terminali kontrol edin.")
            import traceback
            traceback.print_exc()
            # return veya exit yok, pencere açık kalacak
    
    def load_image_as_array(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Uygulamayı Başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = ChangeDetectionApp(root)
    root.mainloop()