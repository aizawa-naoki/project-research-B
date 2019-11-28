1.1_text2vec.py
	bert-as-a-serviceをつかってtextからvectorを作り保存するためのスクリプト
1.2_train_2-layer_network.py
	text2vec.pyを用いて作った文章vectorをaffine2層のネットワークに入力して訓練して検証するスクリプト
	ネットワーク自体はutil.pyにある
2.1_fine-tune.py
	bertを用いてfine-tuningをするためのスクリプト
2.2_evaluation_bert.py
	bertを用いてfine-tuningをした後のmodelのevaluation/printをするスクリプト
2.3_weightedtune.py
	2.1のスクリプトから若干の改変をしたもの
	・誤差を逆伝搬させる際に、誤差に対してラベル{0, 1}の出現頻度を用いて重みづけするようにした。
		⇒正例が少ない属性が存在するため。（1:99程度）
2.4.0_evaluation_attribute_bert.py
	2.2のスクリプトから若干の改変をしたもの
	・2.4.1attribute-tuneはattributeをtokenizerに渡す必要があるため、引数とスクリプトに若干の変更が生じた。
2.4.1_attribute-tune.py
	2.1のスクリプトから若干の改変をしたもの
	・labelのattributeもbert-tokenizerのinputに加えることで"指向的な"判断をできるようにしたもの。
		⇒全体的な正解率を底上げしたかった。
2.4.2_attribute-tune2.py
	2.4.1のスクリプトから若干の改変をしたもの
	・segment maskを付けた