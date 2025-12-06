import optuna
import torch
import torch.optim as optim
from model import ConvVAE, vae_loss
from data_loader import get_dataloaders
from config import Config

def objective(trial):
    # Random seed for reproducibility
    torch.manual_seed(42)
    
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    beta = trial.suggest_float("beta", 1e-6, 1e-2, log=True)
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512])
    
    model = ConvVAE(latent_dim=latent_dim).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    try:
        # Fix: get_dataloaders may return more than 2 values
        dataloaders = get_dataloaders(
            Config.DATA_DIR, 
            batch_size=32, 
            split_ratio=0.2, 
            only_benign=True # Prevent worker crashes
        )
        
        # Handle different return formats
        if len(dataloaders) == 2:
            train_loader, val_loader = dataloaders
        elif len(dataloaders) == 3:
            train_loader, val_loader, _ = dataloaders
        elif len(dataloaders) == 4:
            train_loader, val_loader, _, _ = dataloaders
        else:
            train_loader = dataloaders[0]
            val_loader = dataloaders[1]
    
    except Exception as e:
        print(f"Dataloader hatası: {e}")
        raise optuna.TrialPruned()
    
    # Küçük bir eğitim döngüsü (Örn: Sadece 5 Epoch)
    for epoch in range(5):
        try:
            model.train()
            train_loss = 0
            for images, _ in train_loader:
                images = images.to(Config.DEVICE)
                optimizer.zero_grad()
                recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation kaybını hesapla (ortalama)
            model.eval()
            val_loss = 0
            num_batches = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(Config.DEVICE)
                    recon, mu, logvar = model(images)
                    loss, _, _ = vae_loss(recon, images, mu, logvar, beta)
                    val_loss += loss.item()
                    num_batches += 1
            
            avg_val_loss = val_loss / num_batches if num_batches > 0 else val_loss
            
            # Optuna'ya gidişatı bildir
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        except Exception as e:
            print(f"Epoch {epoch} hatası: {e}")
            raise optuna.TrialPruned()

    return avg_val_loss

if __name__ == '__main__':
    # Multiprocessing için gerekli
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # 3. Optimizasyonu Başlat (SQLite ile kaydet)
    study = optuna.create_study(
        study_name="thyroid_vae_optimization",
        storage="sqlite:///optuna_study.db",
        direction="minimize",
        load_if_exists=True,  # Kesintiden devam edebilir
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    print("Optimizasyon başlıyor... Arkanıza yaslanın.")
    study.optimize(objective, n_trials=20, timeout=3600)  # 1 saat timeout

    # 4. En iyi sonuçları yazdır (başarılı trial kontrolü)
    print("\n" + "="*50)
    
    # Başarılı trial olup olmadığını kontrol et
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) > 0:
        print("En iyi parametreler bulundu:")
        print(study.best_params)
        print(f"\nEn iyi validation loss: {study.best_value:.4f}")
    else:
        print("UYARI: Hiçbir trial başarıyla tamamlanamadı!")
        print(f"Toplam deneme sayısı: {len(study.trials)}")
        print("Lütfen data_loader.py dosyasını kontrol edin ve num_workers=0 olduğundan emin olun.")
    
    print("="*50)

    # 5. Optimizasyon grafiklerini kaydet (opsiyonel)
    if len(completed_trials) > 0:
        try:
            import matplotlib.pyplot as plt
            fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig1.savefig("optimization_history.png")
            fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
            fig2.savefig("param_importances.png")
            print("\nGrafikler kaydedildi: optimization_history.png, param_importances.png")
        except ImportError:
            print("\nGrafik için: pip install matplotlib")