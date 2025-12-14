import torch.nn as nn
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    utils
    )

class StarCoderModel(nn.Module):
    # Class variable to prevent recursive loading
    _loading_from_starvector = False

    def __init__(self, config, **kwargs):
        super(StarCoderModel, self).__init__()
        import os

        # Get the parent model directory if available (passed from model loading)
        parent_model_dir = kwargs.get('parent_model_dir', None)

        # If we're already loading from starvector, don't recurse
        if StarCoderModel._loading_from_starvector:
            raise RuntimeError(
                "Recursive starvector model loading detected. "
                "The starcoder component should not try to load the full starvector model again."
            )

        self.init_tokenizer(config.starcoder_model_name, parent_model_dir=parent_model_dir)

        self.max_length = config.max_length

        # Handle loading config and model with local_files_only when appropriate
        model_load_kwargs = {}
        model_load_kwargs['trust_remote_code'] = True
        model_load_kwargs['torch_dtype'] = config.torch_dtype

        # Check if we're being loaded as part of a starvector model
        # If so, skip loading the starcoder model separately - weights are already loaded
        is_gated_repo = (config.starcoder_model_name == 'bigcode/starcoderbase-1b' or 'bigcode' in config.starcoder_model_name)

        if is_gated_repo and parent_model_dir:
            print(f"[StarVector] Creating starcoder component from config (avoiding gated repo)")
            print(f"[StarVector] Weights will be loaded from starvector model files")

            # Create starcoder config matching the embedded component dimensions
            from transformers import GPTBigCodeConfig

            # Note: n_inner = 8192 (not hidden_size * hidden_size_scale which would be 4096)
            # This was determined by inspecting the actual model weights
            model_config = GPTBigCodeConfig(
                vocab_size=len(self.tokenizer),
                n_positions=config.max_position_embeddings,
                n_embd=config.hidden_size,
                n_layer=config.num_hidden_layers,
                n_head=config.num_attention_heads,
                n_inner=8192,  # Actual dimension from model weights
                activation_function="gelu_pytorch_tanh",
                resid_pdrop=config.dropout,
                embd_pdrop=config.dropout,
                attn_pdrop=config.dropout,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
            )

            if utils.is_flash_attn_2_available() and config.use_flash_attn:
                model_config._attn_implementation = "flash_attention_2"

            # Create model from config (weights loaded by parent starvector model)
            from transformers import GPTBigCodeForCausalLM
            model = GPTBigCodeForCausalLM(config=model_config)

        else:
            # Normal loading path for non-gated repos
            model_load_path = config.starcoder_model_name

            # Load model config
            model_config = AutoConfig.from_pretrained(
                model_load_path,
                trust_remote_code=True
            )

            # Configure special tokens for generation
            model_config.eos_token_id = self.tokenizer.eos_token_id
            model_config.pad_token_id = self.tokenizer.pad_token_id
            model_config.bos_token_id = self.tokenizer.bos_token_id

            if utils.is_flash_attn_2_available():
                model_config.flash_attention = config.use_flash_attn
                model_config._attn_implementation = "flash_attention_2"
            else:
                config.use_flash_attn = False

            # Load the model
            print(f"[StarVector] Loading starcoder model from: {model_load_path}")
            model = AutoModelForCausalLM.from_pretrained(model_load_path, config=model_config, **model_load_kwargs)
        model.resize_token_embeddings(len(self.tokenizer))
        self.transformer = model

        # Prompt the model after image
        self.prompt = '<svg'

    def init_tokenizer(self, model_name, parent_model_dir=None):
        import os

        # Default tokenizer path
        tokenizer_path = model_name

        # If pointing to gated repo, try to find tokenizer locally
        if model_name == 'bigcode/starcoderbase-1b' or 'bigcode' in model_name:
            print(f"[StarVector] Detected gated starcoder repo reference")

            # First, check if parent model directory has tokenizer files
            if parent_model_dir and os.path.isdir(parent_model_dir):
                tokenizer_json = os.path.join(parent_model_dir, 'tokenizer.json')
                if os.path.exists(tokenizer_json):
                    print(f"[StarVector] Found tokenizer files in parent model directory: {parent_model_dir}")
                    tokenizer_path = parent_model_dir
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_path,
                            use_fast=False,
                            local_files_only=True
                        )
                        print(f"[StarVector] Successfully loaded tokenizer from {tokenizer_path}")
                        self._setup_special_tokens()
                        return
                    except Exception as e:
                        print(f"[StarVector] Failed to load from parent dir: {e}")

            # Try to load with local_files_only from cache
            try:
                print(f"[StarVector] Attempting to load tokenizer from HF cache with local_files_only")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    local_files_only=True
                )
                print(f"[StarVector] Successfully loaded tokenizer from cache")
                self._setup_special_tokens()
                return
            except Exception as e:
                print(f"[StarVector] Could not load tokenizer from cache: {e}")

            # If we get here, we couldn't find the tokenizer locally
            raise RuntimeError(
                f"Cannot load tokenizer for {model_name}. "
                f"The bigcode/starcoderbase-1b repository is gated and not accessible. "
                f"The tokenizer files must be available either in the model directory ({parent_model_dir}) "
                f"or in the HuggingFace cache."
            )

        # Load tokenizer for non-gated repos
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        except Exception as e:
            print(f"[StarVector] Standard tokenizer load failed, trying with local_files_only...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, local_files_only=True)

        self._setup_special_tokens()

    def _setup_special_tokens(self):
        # Incude padding and eos tokens in the vocabulary
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})       
        
        self.svg_start_token = "<svg-start>"
        self.image_start_token = "<image-start>"
        self.text_start_token = "<caption-start>"
        
        self.tokenizer.add_tokens([self.svg_start_token, self.image_start_token, self.text_start_token])
        self.svg_start_token_id = self.tokenizer.encode(self.svg_start_token)[0]
