import torch
from snake_game import SnakeGame
from agent import Agent

def play():
    """
    Eğitilmiş modeli test etmek için kullanılan fonksiyon
    """
    # Oyun ve ajan oluştur
    agent = Agent()
    game = SnakeGame()
    
    # Eğitilmiş modeli yükle
    try:
        agent.model.load_state_dict(torch.load('model.pth'))
        agent.model.eval()
        print("Model başarıyla yüklendi!")
    except:
        print("Eğitilmiş model bulunamadı!")
        return
    
    while True:
        # Mevcut durumu al
        state = agent.get_state(game)
        
        # Modelden tahmin al
        final_move = agent.get_action(state)
        
        # Hareketi uygula
        reward, done, score = game.play_step(final_move)
        
        if done:
            # Oyun bitti, yeniden başlat
            game.reset()
            print('Oyun bitti! Skor:', score)

if __name__ == '__main__':
    play()