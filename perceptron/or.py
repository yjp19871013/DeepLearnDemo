import dataset
import logical_gate_linear

train_x, train_labels = dataset.or_load_data_set()
logical_gate_linear.train_logical_gate(train_x, train_labels, "model/or_model.ckpt")