from mmengine.config import read_base

with read_base():
    # from ..alpaca_farm.alpaca_farm_gen_collie_wo_input import alpaca_farm_datasets
    from ..alpaca_farm.alpaca_farm_gen_collie_wo_input import alpaca_farm_no_input_datasets
    from ..alpaca_farm.alpaca_farm_gen_collie_with_input import alpaca_farm_with_input_datasets

  

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
