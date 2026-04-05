from eth_backtesting import build_dataset, Engine, s_dual_momentum_pullback
import warnings
warnings.filterwarnings('ignore')

def debug_losses():
    df = build_dataset()
    phase_a = df[~df['is_oos']].reset_index(drop=True)
    
    sig = s_dual_momentum_pullback(phase_a)
    tr = Engine(phase_a, leverage=3.0).run(sig)
    
    print("TOTAL TRADES:", len(tr))
    losses = tr[tr['pnl'] < 0]
    print(f"LOSSES: {len(losses)}\n")
    
    for i, row in losses.iterrows():
        print(f"Entry: {row['entry_date']} | Exit: {row['exit_date']} | PNL: {row['pnl']*100:.1f}%")

if __name__ == '__main__':
    debug_losses()
