import dataset
import logical_gate_linear

train_x, train_labels = dataset.xor_load_data_set()
logical_gate_linear.train_logical_gate(train_x, train_labels, "model/nand_model.ckpt")