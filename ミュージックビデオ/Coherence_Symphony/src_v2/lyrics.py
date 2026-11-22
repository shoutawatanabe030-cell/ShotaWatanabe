# 歌詞タイミング（Whisper の単語タイムスタンプから作成、秒）と英訳
# 各カード = 同時に表示する行のまとまり。
# 行 = (歌詞, 歌い始め, 歌い終わり, 種別, 英訳)
# 種別: "ja"=日本語(明朝) / "en"=英語(Cinzel) / "bg"=コーラス(括弧つき・小さめ)
# 英訳は日本語の行の下に小さく表示する（英語の行は訳なし = None）
CARDS = [
 # Intro
 [("静寂の奥で　響きはじめる", 9.80, 15.30, "ja", "From deep within the silence, a resonance begins")],
 [("見えない波が　僕らを呼ぶ", 16.90, 22.60, "ja", "An invisible wave is calling us")],
 # Verse 1
 [("バラバラだった　僕らの声", 39.30, 44.70, "ja", "Our voices, once scattered apart")],
 [("すれ違いながら　歩いてきた", 46.30, 53.00, "ja", "We kept walking, passing each other by")],
 [("強がるたびに　遠くなって", 53.80, 59.60, "ja", "Each time we acted strong, we drifted farther away")],
 [("本当の気持ち　隠してた", 60.60, 66.60, "ja", "Hiding what we truly felt")],
 # Pre-Chorus
 [("差し出した手を　握り返せば", 68.20, 75.00, "ja", "When you hold the hand reaching out to you")],
 [("その温もりが　導いてく", 75.60, 80.80, "ja", "Its warmth will lead the way")],
 # Chorus 1
 [("Coherence Symphony", 82.90, 85.80, "en", None), ("手を取り合っていこう", 85.90, 89.30, "ja", "Let's go on, hand in hand")],
 [("We are one", 90.40, 92.05, "en", None), ("ひとつになっていこう", 92.10, 97.30, "ja", "Let's become one")],
 [("Coherence Symphony", 97.60, 100.60, "en", None), ("同じ波動で", 100.66, 105.10, "ja", "On the same vibration")],
 [("We are love", 105.16, 106.75, "en", None), ("もう　ひとりじゃない", 106.80, 109.70, "ja", "You're not alone anymore")],
 # Verse 2
 [("言葉にできない　想いさえも", 113.60, 117.10, "ja", "Even feelings that words cannot express")],
 [("触れた瞬間　伝わっていく", 117.16, 120.85, "ja", "Reach the heart the moment we touch")],
 [("孤独の夜に　震える心", 120.88, 123.95, "ja", "A heart trembling in the lonely night")],
 [("誰かのぬくもり　探してた", 124.30, 128.10, "ja", "Was searching for someone's warmth")],
 # Pre-Chorus 2
 [("揺れてる鼓動　重ねたなら", 128.22, 133.75, "ja", "If we lay our trembling heartbeats together")],
 [("ひとつのリズムに　変わっていく", 134.60, 139.70, "ja", "They turn into a single rhythm")],
 # Chorus 2
 [("Coherence Symphony", 141.40, 143.80, "en", None), ("手を取り合っていこう", 144.29, 147.70, "ja", "Let's go on, hand in hand")],
 [("We are one", 148.97, 150.70, "en", None), ("ひとつになっていこう", 150.72, 155.85, "ja", "Let's become one")],
 [("Coherence Symphony", 155.90, 158.70, "en", None), ("同じ波動で", 159.30, 162.40, "ja", "On the same vibration")],
 [("We are love", 163.53, 165.40, "en", None), ("もう　ひとりじゃない", 165.44, 168.60, "ja", "You're not alone anymore")],
 # Bridge
 [("争うよりも　抱きしめよう", 171.56, 175.15, "ja", "Rather than fight, let's embrace")],
 [("奪うよりも　分け合おう", 175.18, 178.95, "ja", "Rather than take, let's share")],
 [("小さな優しさが　波になって", 179.00, 182.50, "ja", "Small kindnesses become a wave")],
 [("世界を変えていく", 182.52, 186.20, "ja", "And change the world")],
 [("ほら　手を伸ばして", 186.34, 190.65, "ja", "Look, reach out your hand")],
 [("もう　ひとりじゃない", 192.00, 194.40, "ja", "You're not alone anymore")],
 # Final Chorus
 [("(Coherence Symphony)", 200.00, 203.00, "bg", None), ("Coherence Symphony", 203.04, 206.50, "en", None)],
 [("(We are one)", 207.22, 209.38, "bg", None), ("We are one   We are one", 209.40, 211.14, "en", None)],
 [("ひとつになっていこう", 211.16, 214.80, "ja", "Let's become one")],
 [("Coherence Symphony (Hey!)", 214.86, 217.45, "en", None), ("同じ波動で（Hey！）", 217.48, 221.50, "ja", "On the same vibration (Hey!)")],
 [("We are love", 222.40, 224.30, "en", None), ("もう　ひとりじゃない", 224.34, 227.50, "ja", "You're not alone anymore")],
 # Outro
 [("静寂の奥で　また響いてる", 229.24, 235.10, "ja", "From deep within the silence, it resonates again")],
 [("それはきっと　僕らの証", 237.30, 242.60, "ja", "Surely, it is the proof of who we are")],
]
