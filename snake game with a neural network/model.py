import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    """
    Deep Q-Learning için kullanılacak yapay sinir ağı modeli
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Model başlatıcı fonksiyonu
        
        Args:
            input_size: Giriş katmanı boyutu
            hidden_size: Gizli katman boyutu
            output_size: Çıkış katmanı boyutu
        """
        super().__init__()
        
        # Giriş katmanından gizli katmana
        self.linear1 = nn.Linear(input_size, hidden_size)
        
        # Gizli katmandan çıkış katmanına
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def save(self, file_name='model.pth'):
        """
        Modeli kaydetme fonksiyonu
        
        Args:
            file_name: Kaydedilecek dosya adı
        """
        torch.save(self.state_dict(), file_name)
        
    def forward(self, x):
        """
        İleri yayılım fonksiyonu
        
        Args:
            x: Giriş tensörü
            
        Returns:
            Çıkış tensörü
        """
        # İlk katman için ReLU aktivasyonu
        x = F.relu(self.linear1(x))
        
        # Çıkış katmanı
        return self.linear2(x)

class QTrainer:
    """
    Q-Learning eğitim sınıfı
    """
    def __init__(self, model, lr, gamma):
        """
        Eğitici başlatıcı fonksiyonu
        
        Args:
            model: Eğitilecek model
            lr: Öğrenme oranı
            gamma: İndirim faktörü
        """
        self.model = model
        self.lr = lr
        self.gamma = gamma
        
        # Adam optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Mean Squared Error loss
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        """
        Tek bir eğitim adımı
        
        Args:
            state: Mevcut durum
            action: Seçilen aksiyon
            reward: Alınan ödül
            next_state: Sonraki durum
            done: Oyun bitti mi
        """
        # Tensor dönüşümleri
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            # (1, x) şeklinde yeniden boyutlandır
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        # 1. Tahmin edilen Q değerlerini hesapla
        pred = self.model(state)
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        # 2. Q_new = reward + gamma * max(next_predicted Q value)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()