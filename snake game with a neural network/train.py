import torch
import numpy as np
from collections import deque
from snake_game import SnakeGame
from agent import Agent
import matplotlib.pyplot as plt
from IPython import display

def train(load_existing=True):
    """
    Yapay zeka ajanını eğiten ana fonksiyon
    
    Args:
        load_existing: Önceden eğitilmiş modeli yükle
    """
    # Skor geçmişi ve rekor
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    # Oyun ve ajan oluştur
    agent = Agent(load_existing=load_existing)
    game = SnakeGame()
    
    # Eğer önceki model yüklendiyse, başlangıç değerlerini ayarla
    if load_existing and agent.n_games > 0:
        total_score = agent.n_games * 30  # Tahmini ortalama skor
        plot_scores = [30] * agent.n_games  # Geçmiş skorlar için tahmin
        mean_score = total_score / agent.n_games
        plot_mean_scores = [mean_score] * agent.n_games
    
    while True:
        # Mevcut durumu al
        state_old = agent.get_state(game)
        
        # Hareketi belirle
        final_move = agent.get_action(state_old)
        
        # Hareketi uygula ve yeni durumu al
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Kısa bellek eğitimi
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Deneyimi hatırla
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # Oyun bitti, uzun bellek eğitimi
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            # Yeni rekor kontrolü
            if score > record:
                record = score
                agent.model.save()
                
            print('Oyun:', agent.n_games, 'Skor:', score, 'Rekor:', record)
            
            # Grafik için skor hesaplama
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def plot(scores, mean_scores):
    """
    Eğitim sürecini görselleştiren fonksiyon
    
    Args:
        scores: Oyun skorları listesi
        mean_scores: Ortalama skorlar listesi
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Eğitim...')
    plt.xlabel('Oyun Sayısı')
    plt.ylabel('Skor')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def get_user_choice():
    """
    Kullanıcıdan eğitim seçeneğini alır
    
    Returns:
        bool: Var olan modeli kullanma durumu
    """
    while True:
        print("\n=== YILAN OYUNU YAPAY ZEKA EĞİTİM MENÜSÜ ===")
        print("1. Yeni model eğit")
        print("2. Var olan modeli eğitmeye devam et")
        choice = input("\nSeçiminiz (1 veya 2): ")
        
        if choice == "1":
            print("\nYeni model eğitimi başlatılıyor...")
            return False
        elif choice == "2":
            print("\nVar olan model yükleniyor ve eğitime devam ediliyor...")
            return True
        else:
            print("\nHatalı seçim! Lütfen 1 veya 2 girin.")

if __name__ == '__main__':
    use_existing = get_user_choice()
    train(load_existing=use_existing)