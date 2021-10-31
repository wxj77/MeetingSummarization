from collections import defaultdict
"""
All configurations are set following 
"End-to-End Abstractive Summarization for Meetings" paper.
https://arxiv.org/abs/2004.02016
"""

PARAMS = defaultdict(
    # Environment
    device='cuda',
    #device='cpu',
    workers=1,
    gpu_ids=[0],
    data_dir='data/',
    save_dirpath='',
    use_role=True,
    use_pos=False,
    load_pthpath="",
    vocab_word_path='checkpoints/vocab_word',
    # Training Hyperparemter
    batch_size=1,
    num_epochs=30,
    num_epochs_cnn = 5,
    start_eval_epoch=29,
    start_eval_epoch_cnn = 10,
    fintune_word_embedding=True,
    # Transformer
    embedding_size_word=300,
    embedding_size_role=20,
    embedding_size_pos=12,
    num_heads=2,
    num_hidden_layers=2,
    hidden_size=300,
    min_length=100,
    max_length=500,
    gen_max_length=250,
    attention_key_channels=0,
    attention_value_channels=0,
    filter_size=64,
    dropout=0.2,
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.999,
    # Optimizier
    learning_rate=5e-4,
    max_gradient_norm=2,
    # Decoding
    beam_size=12,
    blook_trigram=True,
    model_name="qmsumqueriescnnwithmerge",
    max_batch_numbers_per_epoch = 16,
    do_short_evaluation=False,
    merge_cnn_vocab_into_qm_vocab=True
)