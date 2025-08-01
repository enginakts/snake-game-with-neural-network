import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

# Yön enum sınıfı
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Nokta tuple'ı
Point = namedtuple('Point', 'x, y')

# Renkler
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

# Oyun sabitleri
BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:
    def __init__(self, w=640, h=480):
        """
        Yılan oyunu sınıfının başlatıcı fonksiyonu
        
        Args:
            w: Oyun penceresi genişliği
            h: Oyun penceresi yüksekliği
        """
        self.w = w
        self.h = h
        
        # Pygame başlatma
        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Yılan Oyunu')
        self.clock = pygame.time.Clock()
        
        # Oyunu sıfırla
        self.reset()
    
    def reset(self):
        """Oyunu başlangıç durumuna getirir"""
        # Yılanın başlangıç konumu
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,
            Point(self.head.x-BLOCK_SIZE, self.head.y),
            Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]
        
        # Skor ve yemek
        self.score = 0
        self.food = None
        self._place_food()
        
        # Oyun adım sayısı
        self.frame_iteration = 0
        
    def _place_food(self):
        """Oyun alanına rastgele yemek yerleştirir"""
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        # Eğer yemek yılanın üzerine denk geldiyse tekrar yerleştir
        if self.food in self.snake:
            self._place_food()
    
    def play_step(self, action):
        """
        Oyunun bir adımını simüle eder
        
        Args:
            action: [straight, right_turn, left_turn] şeklinde aksiyon
            
        Returns:
            reward: Ödül değeri
            game_over: Oyunun bitip bitmediği
            score: Güncel skor
        """
        self.frame_iteration += 1
        
        # 1. Kullanıcı girdisini al
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Hareket et
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. Oyun bitti mi kontrol et
        reward = 0
        game_over = False
        
        # Çarpışma veya çok uzun süre yemek yiyememe durumu
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            # Hızlı yeniden başlatma için ekstra bekleme yok
            self.reset()
            return reward, game_over, self.score
        
        # 4. Yemek yeme durumu ve yeni yemek yerleştirme
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Oyun alanını güncelle
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        """
        Verilen noktada çarpışma olup olmadığını kontrol eder
        
        Args:
            pt: Kontrol edilecek nokta (varsayılan: yılan başı)
            
        Returns:
            bool: Çarpışma var mı
        """
        if pt is None:
            pt = self.head
            
        # Duvarlara çarpma
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # Kendine çarpma
        if pt in self.snake[1:]:
            return True
            
        return False
    
    def _update_ui(self):
        """Oyun arayüzünü günceller"""
        self.display.fill(BLACK)
        
        # Yılanı çiz
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Yemeği çiz
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.display.flip()
    
    def _move(self, action):
        """
        Yılanı hareket ettirir
        
        Args:
            action: [straight, right_turn, left_turn] şeklinde aksiyon
        """
        # Action: [straight, right_turn, left_turn]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            # Düz git
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Sağa dön
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            # Sola dön
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)