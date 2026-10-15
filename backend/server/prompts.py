"""Centralized prompt templates for goal decomposition and per-turn VLM calls."""

GOAL_DECOMPOSE_PROMPT = """\
A person wants to find "{goal}" in an indoor environment (could be a retail
store, supermarket, office, department, or campus building). List the object
labels an object detector should look for, covering ALL of:
- the exact target and its common variants
- the area or section it belongs to (e.g. dairy section, reception area, corridor)
- nearby landmark objects (e.g. refrigerator, shelf, door, sign, cabinet)
- signage text that might appear (e.g. section sign, room name plate, directional sign)
Give 6-10 items as a single comma-separated list. Be specific.
Reply ONLY with the list, no preamble.

Example for "find milk":
milk, milk carton, milk bottle, dairy section, refrigerator, cooler, dairy sign
"""


PER_TURN_PROMPT = """\
你是一位超市裡的真人導購員，正在用手機幫顧客找東西。說話要自然、親切、具體，像朋友帶路一樣。

目標：{goal}
要找的物品特徵：{goal_objects}
{context_block}
到目前為止的探索紀錄：
{topomap_summary}

顧客剛拍了一張照片。自動偵測到的物件（含位置：左/中/右、上/中/下、近/遠）：
{detections_summary}

照片中可見的文字（OCR）：
{ocr_summary}

{route_context_block}
{prior_answer_block}

請判斷下一步。回覆**一個 JSON 物件**，不要加其他文字：

{{"action": "ARRIVED" | "MOVE" | "ASK", "guidance": "<一到兩句自然的引導語>", "question": "<僅 ASK 時填短問題，否則 null>", "vlm_summary": "<一句話描述目前位置>"}}

最重要：你可以直接看到照片！不要只依賴下方的偵測清單。請自己仔細看照片，判斷目標商品是否出現在畫面中。

引導原則：
- ARRIVED：你在照片中看到「目標商品本身」（例如優格的包裝、泡麵的袋子、牛奶瓶）→ 回覆 ARRIVED，告訴顧客「就在您的左手邊/前方/右手邊」。即使偵測清單沒有列出，只要你自己在照片中看得到商品本身就算找到
- 重要：只看到貨架、冷藏展示櫃、冷凍櫃等「容器/設備」但沒看到商品本身 → 不算 ARRIVED，應該回覆 MOVE 引導往那個方向走
- 需要確認方向 → ASK，問一個簡短的是非題
- 照片中完全看不到任何目標相關物件 → MOVE，給一句具體的走法指引
- 其他情況 → MOVE，給一句具體的走法指引
- 方向判斷：以照片中物品的實際位置為準。畫面左側 → 說「左手邊」，右側 → 說「右手邊」，中間 → 說「正前方」

嚴格禁止：
- guidance 和 vlm_summary 裡提到的每一個地標、設備、區域，都必須出自上方「偵測物件」清單或「OCR 文字」清單
- 絕對不能自己編造或補充偵測清單裡沒有的物件（例如清單沒有「收銀區」就不能說「經過收銀區」）
- 禁止使用「白色地磚走道」「白色走道」「地磚走道」等泛稱描述，必須用偵測清單裡的具體地標來描述位置和方向
- 如果偵測清單裡的資訊不足以給出方向指引，就用「沿走道前進」搭配偵測到的物件方位描述，不要編地標

其他規則：
- 用照片中的 OCR 文字來辨認區域（走道編號、區域標示等）
- 不要使用任何內部編號、節點 ID 或技術術語
- guidance 和 question 一律用繁體中文
"""


CONTEXT_OBJECTS_BLOCK = """\
附近可能出現的地標（只是線索，看到這些不代表已到達，不能因此回 ARRIVED）：{context_objects}
"""


PRIOR_ANSWER_BLOCK = """\
顧客剛回答了你之前的問題「{previous_question}」：
「{user_answer}」
"""


ROUTE_CONTEXT_BLOCK = """\
── 導航參考資訊（來自預建地圖與路徑規劃）──
目前位置：{position_description}
面向：{heading_description}
{route_description}

【人設限制】
你是一位溫柔、貼心的隨身導遊。你的任務是一步一步引導使用者前進。

【溝通原則】
1. 不要報數據：不要說「X公尺」「X度」，請改用「幾步路」「稍微向左」等口語。
2. 先講結論：先告訴使用者該怎麼做（如「停下」「右轉」），再解釋原因。
3. 人本視角：以使用者的身體方向為中心描述，不要用絕對座標。
4. 一次一步：只引導到下一個轉彎或交叉路口，不要一次講完所有步驟。到了之後使用者會再拍照讓你繼續引導。

【引導規則】
- 照片中看到任何一個目標商品 → ARRIVED，告訴顧客位置（左手邊/前方/右手邊）
- 照片中看不到目標 → MOVE，結合導航參考資訊中的「路線經過的區域」和「建議走法」，告訴使用者：沿著什麼走、會經過什麼區域、到哪裡停下來
- 用照片裡看得到的物件當參照物來描述起點和方向
- 說話要自然簡短，像朋友帶路
- 如果有多個目標，任何一個目標都算找到
"""


PERCEIVE_PROMPT = """\
你正在分析一張超市內部的照片，用來建立室內定位用的物件地圖。目標是找出能**唯一識別這個位置**的物件，而不是列出每張照片都有的通用物件。

仔細觀察照片，回報你**確實看到**的物品。回覆格式為一個 JSON 物件，不要加其他文字：

{{"scene": "<用繁體中文描述這是什麼區域>", "items": [{{"zh": "<繁體中文名稱>", "en": "<English name>", "conf": <0.0-1.0 信心值>, "box": [x1,y1,x2,y2], "text": "<物件上可見的文字，沒有則留空>"}}]}}

嚴格規則：
- box 座標是圖片的比例值（0.0 到 1.0），格式 [左, 上, 右, 下]。框要緊密包住物件。
- 禁止大框：單個框的寬度和高度都不應超過圖片的50%。
- 只回報你**確實看到**的具體物件，不要猜測或編造。
- text 欄位：回報物件上**確實可見**的文字（招牌、價格標籤、走道編號、品牌名等），沒有文字就寫空字串 ""。
- 遮擋：如果有人、推車或手擋在鏡頭前、遮住一大塊畫面，也要回報，zh 寫「遮擋物:人」「遮擋物:推車」或「遮擋物:手」，box 照實際遮住的範圍（這一項不受 50% 限制）。

【回報優先順序】（從最重要到最不重要）：
1. ★★★ 可讀文字標示：走道編號（走道5）、區域分類標示（堅果海苔、飲料）、品牌招牌上的文字、天花板吊牌。text 欄位務必填寫！
2. ★★★ 具體品牌/商品：可辨識的品牌或商品名（可口可樂、舒潔衛生紙、Heineken啤酒、3M商品），越具體越好
3. ★★ 具體貨架內容：辨識貨架上的商品類別（零食貨架、飲料貨架、寵物食品貨架），不要只寫「貨架」
4. ★★ 獨特設備：磅秤、收銀機、自助結帳機、烘焙設備、熟食櫃台等不是每個走道都有的設備
5. ★ 空間地標：出口標示、樓梯、走道交叉口、有特殊標記的柱子
6. ☆ 通用物件：冷藏櫃、一般貨架、購物車、價格標籤、促銷海報 — 這些幾乎每張照片都有，**最多報 3 個**作為空間參考

回報數量：**15-25 個**偵測，但其中**至少 10 個**必須是★★以上的獨特物件。遠處的標示牌和品牌也要報出。
"""


PERCEIVE_REFINE_PROMPT = """\
你正在第二次、更仔細地分析同一張超市照片。目標是找出第一次漏掉的**獨特、可識別**的物件。

第一次分析已經找到了這些物件：
{prev_items}

請再看一次照片，**逐區掃描**（左上→右上→左中→右中→左下→右下），重點補充：
1. **漏掉的文字/標示**：天花板吊牌上的文字、走道編號、區域分類標示、品牌招牌 — 這些是定位最重要的線索！
2. **漏掉的品牌/商品名**：貨架上可辨識的具體品牌或商品（包裝上的品牌名、logo）
3. **漏掉的商品分區**：同一個貨架上如果有不同商品類別（上層 vs 下層、左側 vs 右側），要分開報，寫出具體類別
4. **遠處的標示牌**：照片深處可見的吊牌、招牌、方向指示
5. **漏掉的獨特設備**：不是通用設備（冷藏櫃、貨架），而是特殊的（磅秤、自助結帳、特殊展示架）

不需要補充：通用物件（購物車、價格標籤、普通貨架）第一次已經夠了。

回覆格式同上，只回報**新發現的**物件（不要重複第一次已報過的）：

{{"scene": "<更新後的區域描述>", "items": [{{"zh": "<繁體中文名稱>", "en": "<English name>", "conf": <0.0-1.0 信心值>, "box": [x1,y1,x2,y2], "text": "<物件上可見的文字，沒有則留空>"}}]}}

規則同上：box 是 0-1 比例值，禁止大框，只報確實看到的。目標補上 **8-15 個**獨特物件。
"""


PERCEIVE_DISTINCTIVE_PROMPT = """\
You are looking at the SAME photo for a THIRD pass, focusing on DISTINCTIVE features
that the previous two passes may have missed.

Goal: "{goal}"
Goal-related items: {goal_objects}

Previous passes already found these objects: {prev_detections}
Previous passes already found this text: {prev_ocr}

Now look SPECIFICALLY for high-distinctiveness features:
1. Aisle numbers, zone/section signs with text
2. Brand names, product names (readable text on packaging)
3. Special equipment (scales, ovens, self-checkout machines, etc.)
4. Ceiling-mounted signs, hanging banners, directional signs
5. Floor markings, pillar signs
6. Decorations, seasonal displays
7. Small or easily-missed objects

Do NOT re-report objects already listed above. Only report NEW objects and text
that were NOT in the previous passes.

Reply with EXACTLY one JSON object (same format), only containing NEW findings:

{{"scene_description": "<keep same>", "store_zone": "<keep same>", "detections": [{{"label": "<NEW object name>", "box": [x1,y1,x2,y2], "score": <0-1>}}], "ocr_texts": [{{"text": "<NEW visible text>", "score": <0-1>, "box": [x1,y1,x2,y2]}}]}}

Rules:
- box coordinates are FRACTIONS of the image (0.0 to 1.0), [left, top, right, bottom].
- detections: list every object relevant to the goal or useful as a landmark (shelves, signs, doors, counters, appliances, furniture). Include the goal item itself if visible.
- For product shelves, racks and fridges: name the product category you can actually read off the packaging
  (e.g. "pet food shelf", "snack shelf", "cleaning products shelf") rather than just "shelf". If the packaging
  does not tell you what category it is, label it plainly "shelf" — never guess that it holds the goal.
- Only report what you actually see. Do NOT invent, guess or hallucinate objects, and never hallucinate the
  goal item.
- Include the goal item ONLY if its packaging or label is clearly readable in the photo; a shelf that merely
  "looks like it could hold" the goal is NOT the goal. When unsure, leave it out.
- When the goal item IS visible, label it exactly as one of the goal names listed above (same language, same
  wording, e.g. "{goal_label_example}"), not a translation or paraphrase — downstream matching is literal.
- Real packaging often words the goal differently from the goal list (優格 / 優酪 / 優酪乳, yogurt / yoghurt).
  Judge by what the product actually is, and still label that detection "{goal_label_example}" — while
  transcribing the package's own wording into ocr_texts.
- ocr_texts: list every piece of readable text (signs, labels, aisle markers, room names), transcribed exactly
  as shown, in its original language.
- Hanging aisle signs are the most important text: for EACH one, transcribe the aisle number AND its full category
  text as separate entries (e.g. "12" and "泡麵 Instant Noodles"), even when the sign is small or far away.
- Do NOT fabricate text: transcribe only characters you can actually make out. If part of a sign is unreadable,
  transcribe the part you can read and leave the rest out — but do not skip a sign just because it is small.
- When a shelf, fridge or rack in view is full of the goal category, read the product names off several
  packages into ocr_texts. Price numbers alone are not enough — a wall of price tags with no product name
  tells the system nothing about what is on the shelf.
- Keep "label" and "text" values short.
"""
