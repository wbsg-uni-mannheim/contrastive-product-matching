import torch
from torch import nn

from transformers import AutoModel, AutoConfig

from pdb import set_trace

def init_weights(module, init_type='xavier'):
    """Initialize the weights"""
    if init_type =='default':
        return
    elif init_type == 'huggingface':
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    elif init_type == 'kaiming':
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.01)
                #module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    elif init_type == 'xavier':
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.01)
                #module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mean_pooling_multidimensional(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, -2) / torch.clamp(input_mask_expanded.sum(-2), min=1e-9)

class HttCta(nn.Module):

    def __init__(self, pretrained_path=False, pos_neg_ratio=None, frozen=False, pool=True, sum_axial=True, use_colcls=True, gradient_checkpointing=False):
        super().__init__()

        self.pos_neg_ratio = pos_neg_ratio
        self.pretrained_path = pretrained_path
        self.frozen = frozen
        self.pool = pool
        self.sum_axial = sum_axial
        self.use_colcls = use_colcls
        self.gradient_checkpointing = gradient_checkpointing

        self.table_model = TableModel(self.frozen, self.pool, self.sum_axial, self.gradient_checkpointing)
        
        self.config = self.table_model.cell_encoder.config
        self.cls = HTTClassificationHead(self.config, self.use_colcls)

        if pretrained_path:
            checkpoint = torch.load(self.pretrained_path)
            self.load_state_dict(checkpoint, strict=False)


    def forward(self, table_input_ids, table_attention_mask, table_mv, header_input_ids, header_attention_mask, header_mv, meta_input_ids, meta_attention_mask, meta_mv, padded_rows, padded_cols, cta_labels):
        
        outputs, attention_masks = self.table_model(table_input_ids, table_attention_mask, table_mv, header_input_ids, header_attention_mask, header_mv, meta_input_ids, meta_attention_mask, meta_mv, padded_rows, padded_cols)

        logits = self.cls(outputs, attention_masks)
        loss = None
        
        if self.pos_neg_ratio is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_neg_ratio.clone().to(logits.device))
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        active_loss = (cta_labels != -100)
        # active_logits = logits[active_loss]
        # active_labels = labels[active_loss]
        #loss = loss_fct(active_logits, active_labels.float())
        loss = loss_fct(logits, cta_labels.float())
        active_labels = cta_labels[active_loss].numel()
        loss = loss*active_loss.float()
        loss = torch.sum(loss)/active_labels
        
        fct = nn.Sigmoid()
        logits = torch.nan_to_num(logits, nan=-10.0)
        logits = fct(logits)

        output = (logits, cta_labels)
        return ((loss,) + output) if loss is not None else output

class HttCorruptionPretraining(nn.Module):

    def __init__(self, pos_neg_ratio=None, frozen=False, pool=True, sum_axial=True, gradient_checkpointing=False):
        super().__init__()

        self.pos_neg_ratio = pos_neg_ratio

        self.frozen = frozen
        self.pool = pool
        self.sum_axial = sum_axial
        self.gradient_checkpointing = gradient_checkpointing

        self.table_model = TableModel(self.frozen, self.pool, self.sum_axial, self.gradient_checkpointing)
        
        self.config = self.table_model.cell_encoder.config
        self.classifier = DiscriminatorHead(self.config)

    def forward(self, table_input_ids, table_attention_mask, table_mv, header_input_ids, header_attention_mask, header_mv, meta_input_ids, meta_attention_mask, meta_mv, padded_rows, padded_cols, corr_labels):
        
        outputs, _ = self.table_model(table_input_ids, table_attention_mask, table_mv, header_input_ids, header_attention_mask, header_mv, meta_input_ids, meta_attention_mask, meta_mv, padded_rows, padded_cols)
        logits = self.classifier(outputs)
        loss = None
        
        if self.pos_neg_ratio is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.full([1], self.pos_neg_ratio).to(logits.device))
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        active_loss = (corr_labels != -100)
        # active_logits = logits[active_loss]
        # active_labels = labels[active_loss]
        # loss = loss_fct(active_logits, active_labels.float())
        logits = logits[:,2:,2:,:]
        loss = loss_fct(logits, corr_labels.float())
        active_labels = corr_labels[active_loss].numel()
        loss = loss*active_loss.float()
        loss = torch.sum(loss)/active_labels

        fct = nn.Sigmoid()
        logits = torch.nan_to_num(logits, nan=-10.0)
        logits = fct(logits)
        
        output = (logits, corr_labels)
        return ((loss,) + output) if loss is not None else output

class TableModel(nn.Module):
    def __init__(self, frozen=False, pool=True, sum_axial=True, gradient_checkpointing=False):
        super().__init__()

        self.pool = pool
        config = AutoConfig.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', gradient_checkpointing=gradient_checkpointing)
        self.cell_encoder = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', config=config)
        #self.cell_encoder = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        
        self.type_embeddings = TypeEmbeddings(self.cell_encoder.config)
        self.pos_emb = AxialPositionalEmbedding(self.cell_encoder.config, dim = self.cell_encoder.config.hidden_size, shape = (512, 512), emb_dim_index = 3)
        
        self.LayerNorm = nn.LayerNorm(self.cell_encoder.config.hidden_size, eps=self.cell_encoder.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.cell_encoder.config.hidden_dropout_prob)
        
        axial_encoder = HTTEncoder(self.cell_encoder.config, sum_axial, gradient_checkpointing)
        
        self.axial_encoder = axial_encoder

        if self.pool:
            params = [self.cell_encoder.pooler.dense.weight, self.cell_encoder.pooler.dense.bias]
            for param in params:
                param.requires_grad=False
        
        if frozen:
            for param in self.cell_encoder.parameters():
                param.requires_grad = False
        
        init_weights(self.LayerNorm)
            
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0], inputs[1])
            return inputs
        return custom_forward
    
    def forward(self, table_input_ids, table_attention_mask, table_mv, header_input_ids, header_attention_mask, header_mv, meta_input_ids, meta_attention_mask, meta_mv, padded_rows, padded_cols):
        
        cls_col = self.type_embeddings(None, 3)
        cls_row = self.type_embeddings(None, 4)
        cls_table = self.type_embeddings(None, 5)
        pad = self.type_embeddings(None, 7)

        if self.pool:
            output = self.cell_encoder(table_input_ids, table_attention_mask)
            output = mean_pooling_multidimensional(output, table_attention_mask)
        else:
            output = self.cell_encoder(table_input_ids, table_attention_mask)['pooler_output']

        table_stacked_embed = self.type_embeddings(output, 0)

        if self.pool:
            header_encoded = self.cell_encoder(header_input_ids, header_attention_mask)
            header_encoded = mean_pooling_multidimensional(header_encoded, header_attention_mask)
        else:
            header_encoded = self.cell_encoder(header_input_ids, header_attention_mask)['pooler_output']
        
        header_type_embed = self.type_embeddings(header_encoded, 1)
        
        if self.pool:
            metadata_encoded = self.cell_encoder(meta_input_ids, meta_attention_mask)
            metadata_encoded = mean_pooling_multidimensional(metadata_encoded, meta_attention_mask)
        else:
            metadata_encoded = self.cell_encoder(meta_input_ids, meta_attention_mask)['pooler_output']
        
        metadata_type_embed = self.type_embeddings(metadata_encoded, 2)
        
        attention_mask = table_mv
        table_stacked_embed = torch.cat((header_type_embed.unsqueeze(1), table_stacked_embed), 1)
        attention_mask = torch.cat((header_mv.unsqueeze(1), attention_mask), 1)
        
        # start building attention mask to not consider MV or padding
        attention_mask[attention_mask==1] = -10000
        attention_mask[attention_mask!=-10000] = 0
        
        cls_col_embed = cls_col.repeat(table_stacked_embed.shape[0], 1, table_stacked_embed.shape[2], 1)
        
        table_stacked_embed = torch.cat((cls_col_embed, table_stacked_embed), 1)
        attention_mask = torch.cat((padded_cols.unsqueeze(1), attention_mask), 1)
        
        cls_row_single = cls_row.repeat(table_stacked_embed.shape[0], table_stacked_embed.shape[1], 1, 1)
        cls_row_single[:,0,0,:] = cls_table.squeeze(0)
        
        table_stacked_embed = torch.cat((cls_row_single, table_stacked_embed), 2)

        mod_padded_rows = torch.cat((torch.full((padded_rows.shape[0],2), 0).to(padded_rows.device), padded_rows), 1)
        attention_mask = torch.cat((mod_padded_rows.unsqueeze(2), attention_mask), 2)
        
        table_stacked_embed = self.pos_emb(table_stacked_embed)
        
        metadata_type_embed = metadata_type_embed.unsqueeze(1).unsqueeze(1)
        metadata_type_col = metadata_type_embed.repeat(1, 1, table_stacked_embed.shape[2], 1)

        table_stacked_embed = torch.cat((metadata_type_col, table_stacked_embed), 1)

        meta_mv[meta_mv==1] = -10000
        meta_mv_col = meta_mv.unsqueeze(1).unsqueeze(1)
        meta_mv_col = meta_mv_col.repeat(1,*metadata_type_col.shape[1:-1])

        mod_padded_cols = torch.cat((torch.full((padded_cols.shape[0],1), 0).to(padded_cols.device), padded_cols), 1).unsqueeze(1)
        meta_mv_col[mod_padded_cols==-10000] = -10000
        attention_mask = torch.cat((meta_mv_col, attention_mask), 1)
        
        metadata_type_row = metadata_type_embed.repeat(1, table_stacked_embed.shape[1], 1, 1)
        metadata_type_row[:,0,0,:] = pad.squeeze(0)
        
        table_stacked_embed = torch.cat((metadata_type_row, table_stacked_embed), 2)

        meta_mv_row = meta_mv.unsqueeze(1).unsqueeze(1)
        meta_mv_row = meta_mv_row.repeat(1,*metadata_type_row.shape[1:-1])
        
        mod_padded_rows = torch.cat((torch.full((padded_rows.shape[0],3), 0).to(padded_rows.device), padded_rows), 1)
        meta_mv_row[mod_padded_rows==-10000] = -10000
        attention_mask = torch.cat((meta_mv_row, attention_mask), 2)

        attention_mask[:,0,0] = -10000

        tables_batch = table_stacked_embed
        attention_masks_batch = attention_mask.unsqueeze(-1)

        tables_batch = self.dropout(tables_batch)
        tables_batch = self.LayerNorm(tables_batch)

        tables_batch = self.axial_encoder(tables_batch, attention_masks_batch)

        return tables_batch, attention_masks_batch

class HttForRe(nn.Module):

    def __init__(self, pretrained_path=False, pos_neg_ratio=None, frozen=False, pool=True, sum_axial=True, use_colcls=True, gradient_checkpointing=False, num_labels=121):
        super().__init__()

        self.pos_neg_ratio = pos_neg_ratio
        self.pretrained_path = pretrained_path
        self.frozen = frozen
        self.pool = pool
        self.sum_axial = sum_axial
        self.use_colcls = use_colcls
        self.gradient_checkpointing = gradient_checkpointing
        self.num_labels = num_labels

        self.table_model = TableModelForCta(self.frozen, self.pool, self.sum_axial, self.num_labels)
        
        self.config = self.table_model.cell_encoder.config
        self.cls = REClassificationHead(self.config, self.use_colcls)

        if pretrained_path:
            checkpoint = torch.load(self.pretrained_path)
            self.load_state_dict(checkpoint, strict=False)


    def forward(self, tables):

        outputs, labels, attention_masks = self.table_model(tables)

        logits = self.cls(outputs, attention_masks)
        loss = None
        
        if self.pos_neg_ratio is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_neg_ratio.clone().to(logits.device))
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        active_loss = (labels != -100)
        # active_logits = logits[active_loss]
        # active_labels = labels[active_loss]
        #loss = loss_fct(active_logits, active_labels.float())
        loss = loss_fct(logits, labels.float())
        active_labels = labels[active_loss].numel()
        loss = loss*active_loss.float()
        loss = torch.sum(loss)/active_labels

        fct = nn.Sigmoid()
        logits = torch.nan_to_num(logits, nan=-10.0)
        logits = fct(logits)
        
        output = (logits, labels)
        return ((loss,) + output) if loss is not None else output

class HttForEl(nn.Module):

    def __init__(self, pretrained_path=False, frozen=False, pool=True, sum_axial=True, gradient_checkpointing=False):
        super().__init__()

        self.pretrained_path = pretrained_path
        self.frozen = frozen
        self.pool = pool
        self.sum_axial = sum_axial
        self.gradient_checkpointing = gradient_checkpointing

        self.table_model = TableModelForEl(self.frozen, self.pool, self.sum_axial)
        
        self.config = self.table_model.cell_encoder.config
        self.cls = ElClassificationHead(self.config)

        if pretrained_path:
            checkpoint = torch.load(self.pretrained_path)
            self.load_state_dict(checkpoint, strict=False)


    def forward(self, tables):

        outputs, labels, candidates, cand_masks = self.table_model(tables)

        scores = self.cls(outputs, candidates)

        cand_masks[cand_masks==1] = -10000
        scores += cand_masks.unsqueeze(1)

        loss = None
        
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(scores.transpose(1,2), labels.view(labels.shape[0], -1))

        fct = nn.Softmax(-1)
        scores = torch.nan_to_num(scores, nan=-10000.0)

        scores = fct(scores)

        scores = torch.argmax(scores, axis=-1)
        
        output = (scores, labels.view(labels.shape[0], -1))
        return ((loss,) + output) if loss is not None else output

class HttForCta(nn.Module):

    def __init__(self, pretrained_path=False, pos_neg_ratio=None, frozen=False, pool=True, sum_axial=True, use_colcls=True, gradient_checkpointing=False, num_labels=255):
        super().__init__()

        self.pos_neg_ratio = pos_neg_ratio
        self.pretrained_path = pretrained_path
        self.frozen = frozen
        self.pool = pool
        self.sum_axial = sum_axial
        self.use_colcls = use_colcls
        self.gradient_checkpointing = gradient_checkpointing
        self.num_labels = num_labels
        

        self.table_model = TableModelForCta(self.frozen, self.pool, self.sum_axial, self.num_labels)
        
        self.config = self.table_model.cell_encoder.config
        self.cls = HTTClassificationHead(self.config, self.use_colcls)

        if pretrained_path:
            checkpoint = torch.load(self.pretrained_path)
            self.load_state_dict(checkpoint, strict=False)


    def forward(self, tables):

        outputs, labels, attention_masks = self.table_model(tables)

        logits = self.cls(outputs, attention_masks)
        loss = None
        
        if self.pos_neg_ratio is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_neg_ratio.clone().to(logits.device))
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        active_loss = (labels != -100)
        # active_logits = logits[active_loss]
        # active_labels = labels[active_loss]
        #loss = loss_fct(active_logits, active_labels.float())
        loss = loss_fct(logits, labels.float())
        active_labels = labels[active_loss].numel()
        loss = loss*active_loss.float()
        loss = torch.sum(loss)/active_labels

        fct = nn.Sigmoid()
        logits = torch.nan_to_num(logits, nan=-10.0)
        logits = fct(logits)
        
        output = (logits, labels)
        return ((loss,) + output) if loss is not None else output

class TableModelForPreTraining(nn.Module):
    def __init__(self, frozen=False, pool=True, sum_axial=True):
        super().__init__()
        
        self.pool = pool
        # config = AutoConfig.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', gradient_checkpointing=True)
        # self.cell_encoder = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', config=config)
        self.cell_encoder = AutoModelOriginal.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        
#         cell_encoder = BertModel.from_pretrained('bert-base-uncased')
#         cell_encoder.add_adapter("cell_summarization")
#         cell_encoder.train_adapter("cell_summarization")
#         cell_encoder.set_active_adapters("cell_summarization")
#         self.cell_encoder = cell_encoder
        
#         for param in self.cell_encoder.parameters():
#             param.requires_grad = False
        
        self.type_embeddings = TypeEmbeddings(self.cell_encoder.config)
        self.pos_emb = AxialPositionalEmbedding(self.cell_encoder.config, dim = self.cell_encoder.config.hidden_size, shape = (512, 512), emb_dim_index = 3)
        
        self.LayerNorm = nn.LayerNorm(self.cell_encoder.config.hidden_size, eps=self.cell_encoder.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.cell_encoder.config.hidden_dropout_prob)
        
        axial_encoder = HTTEncoder(self.cell_encoder.config, sum_axial)
        
        self.axial_encoder = axial_encoder

        if frozen:
            for param in self.cell_encoder.parameters():
                param.requires_grad = False
        
        init_weights(self.LayerNorm)
            
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0], inputs[1])
            return inputs
        return custom_forward
    
    def forward(self, tables):
        
        max_cols = -1
        max_rows = -1
        
        
        table_tensors = []
        labels = []
        attention_masks = []
        
        cls_col = self.type_embeddings(None, 3)
        cls_row = self.type_embeddings(None, 4)
        cls_table = self.type_embeddings(None, 5)
        pad = self.type_embeddings(None, 7)
        
        for [[cols, cells_corrupted_idx, cell_labels], [headers, headers_corrupted_idx, headers_labels], [metadata, metadata_missing]] in tables:
            
            col_tensors = []
            
            for col in cols:
                if self.pool:
                    output = self.cell_encoder(col['input_ids'], col['attention_mask'])
                    output = mean_pooling(output, col['attention_mask'])
                else:
                    output = self.cell_encoder(col['input_ids'], col['attention_mask'])['pooler_output']
                
                col_tensors.append(output)

            if self.pool:
                header_encoded = self.cell_encoder(headers['input_ids'], headers['attention_mask'])
                header_encoded = mean_pooling(header_encoded, headers['attention_mask'])
            else:
                header_encoded = self.cell_encoder(headers['input_ids'], headers['attention_mask'])['pooler_output']
            
            
            header_encoded = header_encoded[headers_corrupted_idx,:]
            header_type_embed = self.type_embeddings(header_encoded, 1)
            
            if self.pool:
                metadata_encoded = self.cell_encoder(metadata['input_ids'], metadata['attention_mask'])
                metadata_encoded = mean_pooling(metadata_encoded, metadata['attention_mask'])
            else:
                metadata_encoded = self.cell_encoder(metadata['input_ids'], metadata['attention_mask'])['pooler_output']
            
            metadata_type_embed = self.type_embeddings(metadata_encoded, 2)
            
            # TODO check stacking is done correctly!
            table_stacked = torch.stack(col_tensors, dim=1)
            
            table_stacked_shaped = table_stacked.reshape(-1,table_stacked.shape[-1])
            cells_corrupted_idx_shaped = cells_corrupted_idx.reshape(-1)
            table_stacked_shaped = table_stacked_shaped[cells_corrupted_idx_shaped,:]
            table_stacked_shaped = table_stacked_shaped.reshape(table_stacked.shape)
            table_stacked_embed = self.type_embeddings(table_stacked_shaped, 0)   
            
            table_stacked_embed = torch.cat((header_type_embed.unsqueeze(0), table_stacked_embed), 0)         
            labels_stacked = torch.cat((headers_labels.unsqueeze(0), cell_labels))
            
            # start building attention mask to not consider MV or padding
            attention_mask = labels_stacked.clone()
            attention_mask[attention_mask==-100] = -10000
            attention_mask[attention_mask!=-10000] = 0
            
            
            col_size = table_stacked_embed.shape[1]
            row_size = table_stacked_embed.shape[0]
            
            cls_col_embed = cls_col.repeat(1, col_size, 1)
            
            table_stacked_embed = torch.cat((cls_col_embed, table_stacked_embed), 0)
            labels_stacked = torch.cat((torch.full(cls_col_embed.shape[:-1], -100).to(labels_stacked.device), labels_stacked), 0)
            attention_mask = torch.cat((torch.full(cls_col_embed.shape[:-1], 0).to(attention_mask.device), attention_mask), 0)
            
            cls_row_single = cls_row.repeat(row_size+1, 1, 1)
            cls_row_single[0,0,:] = cls_table.squeeze(0)
            
            table_stacked_embed = torch.cat((cls_row_single, table_stacked_embed), 1)
            labels_stacked = torch.cat((torch.full(cls_row_single.shape[:-1], -100).to(labels_stacked.device), labels_stacked), 1)
            attention_mask = torch.cat((torch.full(cls_row_single.shape[:-1], 0).to(attention_mask.device), attention_mask), 1)
            
            table_stacked_embed = self.pos_emb(table_stacked_embed)
               
            col_size += 1
            row_size += 1
            
            metadata_type_col = metadata_type_embed.repeat(1, col_size, 1)
            
            table_stacked_embed = torch.cat((metadata_type_col, table_stacked_embed), 0)          
            labels_stacked = torch.cat((torch.full(metadata_type_col.shape[:-1], -100).to(labels_stacked.device), labels_stacked), 0)
            attention_mask = torch.cat((torch.full(metadata_type_col.shape[:-1], -10000 if metadata_missing else 0).to(attention_mask.device), attention_mask), 0) 
            
            metadata_type_row = metadata_type_embed.repeat(row_size+1, 1, 1)
            metadata_type_row[0,0,:] = pad.squeeze(0)
            
            table_stacked_embed = torch.cat((metadata_type_row, table_stacked_embed), 1)
            labels_stacked = torch.cat((torch.full(metadata_type_row.shape[:-1], -100).to(labels_stacked.device), labels_stacked), 1)
            attention_mask = torch.cat((torch.full(metadata_type_row.shape[:-1], -10000 if metadata_missing else 0).to(attention_mask.device), attention_mask), 1) 
            attention_mask[0,0] = -10000
            
            col_size += 1
            row_size += 1
            
            if col_size > max_cols:
                max_cols = col_size
            if row_size > max_rows:
                max_rows = row_size
            
            table_tensors.append(table_stacked_embed)
            labels.append(labels_stacked)
            attention_masks.append(attention_mask)
        
        for i, table_tensor in enumerate(table_tensors):
            
            cur_cols = table_tensor.shape[1]
            cur_rows = table_tensor.shape[0]
            cur_labels = labels[i]
            cur_attention_mask = attention_masks[i]
            
            if cur_cols < max_cols:
                col_pad = pad.repeat(cur_rows, max_cols-cur_cols, 1)
                table_tensor = torch.cat((table_tensor, col_pad), 1)
                cur_labels = torch.cat((cur_labels, torch.full(col_pad.shape[:-1], -100).to(cur_labels.device)), 1)
                cur_attention_mask = torch.cat((cur_attention_mask, torch.full(col_pad.shape[:-1], -10000).to(cur_attention_mask.device)), 1)
                
            if cur_rows < max_rows:
                row_pad = pad.repeat(max_rows-cur_rows, max_cols, 1)
                table_tensor = torch.cat((table_tensor, row_pad), 0)
                cur_labels = torch.cat((cur_labels, torch.full(row_pad.shape[:-1], -100).to(cur_labels.device)), 0)
                cur_attention_mask = torch.cat((cur_attention_mask, torch.full(row_pad.shape[:-1], -10000).to(cur_attention_mask.device)), 0)
                
            table_tensors[i] = table_tensor
            cur_labels = cur_labels.reshape(cur_labels.shape[0], cur_labels.shape[1], -1)
            labels[i] = cur_labels
            
            cur_attention_mask = cur_attention_mask.reshape(cur_attention_mask.shape[0], cur_attention_mask.shape[1], -1)
            attention_masks[i] = cur_attention_mask

        tables_batch = torch.stack(table_tensors)
        labels_batch = torch.stack(labels)
        attention_masks_batch = torch.stack(attention_masks)
        
        tables_batch = self.dropout(tables_batch)
        tables_batch = self.LayerNorm(tables_batch)
        
#         for i, layer_module in enumerate(self.axial_encoder):
#             if self.cell_encoder.config.gradient_checkpointing:
#                 tables_batch = checkpoint.checkpoint(self.custom(layer_module), tables_batch, attention_masks_batch)
#             else:
#                 tables_batch = layer_module(tables_batch, attention_masks_batch)

        tables_batch = self.axial_encoder(tables_batch, attention_masks_batch)

        return tables_batch, labels_batch

class TableModelForEl(nn.Module):
    def __init__(self, frozen=False, pool=True, sum_axial=True):
        super().__init__()

        self.pool = pool
        #config = AutoConfig.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', gradient_checkpointing=True)
        #self.cell_encoder = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', config=config)
        self.cell_encoder = AutoModelOriginal.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        
        self.type_embeddings = TypeEmbeddings(self.cell_encoder.config)
        self.pos_emb = AxialPositionalEmbedding(self.cell_encoder.config, dim = self.cell_encoder.config.hidden_size, shape = (512, 512), emb_dim_index = 3)
        
        self.LayerNorm = nn.LayerNorm(self.cell_encoder.config.hidden_size, eps=self.cell_encoder.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.cell_encoder.config.hidden_dropout_prob)
        
        axial_encoder = HTTEncoder(self.cell_encoder.config, sum_axial)
        
        self.axial_encoder = axial_encoder

        if frozen:
            for param in self.cell_encoder.parameters():
                param.requires_grad = False
        
        init_weights(self.LayerNorm)
            
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0], inputs[1])
            return inputs
        return custom_forward
    
    def forward(self, tables):
        
        max_cols = -1
        max_rows = -1

        max_cand = -1
        
        table_tensors = []
        labels = []
        attention_masks = []
        candidates = []
        cand_masks = []
        
        cls_col = self.type_embeddings(None, 3)
        cls_row = self.type_embeddings(None, 4)
        cls_table = self.type_embeddings(None, 5)
        pad = self.type_embeddings(None, 7)
        
        for [[cols, cells_missing], [headers, headers_missing], [metadata, metadata_missing], [label_table, cand]] in tables:
            
            col_tensors = []
            
            for col in cols:
                if self.pool:
                    output = self.cell_encoder(col['input_ids'], col['attention_mask'])
                    output = mean_pooling(output, col['attention_mask'])
                else:
                    output = self.cell_encoder(col['input_ids'], col['attention_mask'])['pooler_output']
                col_tensors.append(output)

            if self.pool:
                header_encoded = self.cell_encoder(headers['input_ids'], headers['attention_mask'])
                header_encoded = mean_pooling(header_encoded, headers['attention_mask'])
            else:
                header_encoded = self.cell_encoder(headers['input_ids'], headers['attention_mask'])['pooler_output']
            
            header_type_embed = self.type_embeddings(header_encoded, 1)

            if self.pool:
                candidates_encoded = self.cell_encoder(cand['input_ids'], cand['attention_mask'])
                candidates_encoded = mean_pooling(candidates_encoded, cand['attention_mask'])
            else:
                candidates_encoded = self.cell_encoder(cand['input_ids'], cand['attention_mask'])['pooler_output']

            if candidates_encoded.shape[0] > max_cand:
                max_cand = candidates_encoded.shape[0]

            if self.pool:
                metadata_encoded = self.cell_encoder(metadata['input_ids'], metadata['attention_mask'])
                metadata_encoded = mean_pooling(metadata_encoded, metadata['attention_mask'])
            else:
                metadata_encoded = self.cell_encoder(metadata['input_ids'], metadata['attention_mask'])['pooler_output']
            
            metadata_type_embed = self.type_embeddings(metadata_encoded, 2)

            # TODO check stacking is done correctly!
            table_stacked = torch.stack(col_tensors, dim=1)
            
            table_stacked_embed = self.type_embeddings(table_stacked, 0)   
            
            attention_mask = cells_missing

            table_stacked_embed = torch.cat((header_type_embed.unsqueeze(0), table_stacked_embed), 0)
            attention_mask = torch.cat((headers_missing.unsqueeze(0), attention_mask), 0)

            labels_stacked = torch.cat((torch.full((header_type_embed.shape[0],),-100).unsqueeze(0).to(header_type_embed.device) , label_table),0) 
            
            #TODO BUILD attention mask based on MV
            # start building attention mask to not consider MV or padding
            attention_mask[attention_mask==1] = -10000
            attention_mask[attention_mask!=-10000] = 0
            
            
            col_size = table_stacked_embed.shape[1]
            row_size = table_stacked_embed.shape[0]
            
            cls_col_embed = cls_col.repeat(1, col_size, 1)
            
            table_stacked_embed = torch.cat((cls_col_embed, table_stacked_embed), 0)
            labels_stacked = torch.cat((torch.full(cls_col_embed.shape[:-1], -100).to(labels_stacked.device), labels_stacked), 0)
            attention_mask = torch.cat((torch.full(cls_col_embed.shape[:-1], 0).to(attention_mask.device), attention_mask), 0)
            
            cls_row_single = cls_row.repeat(row_size+1, 1, 1)
            cls_row_single[0,0,:] = cls_table.squeeze(0)
            
            table_stacked_embed = torch.cat((cls_row_single, table_stacked_embed), 1)

            labels_stacked = torch.cat((torch.full(cls_row_single.shape[:-1], -100).to(labels_stacked.device), labels_stacked), 1)

            attention_mask = torch.cat((torch.full(cls_row_single.shape[:-1], 0).to(attention_mask.device), attention_mask), 1)
            
            table_stacked_embed = self.pos_emb(table_stacked_embed)
               
            col_size += 1
            row_size += 1
            
            metadata_type_col = metadata_type_embed.repeat(1, col_size, 1)
            
            table_stacked_embed = torch.cat((metadata_type_col, table_stacked_embed), 0)
            labels_stacked = torch.cat((torch.full(metadata_type_col.shape[:-1], -100).to(labels_stacked.device), labels_stacked), 0)
            attention_mask = torch.cat((torch.full(metadata_type_col.shape[:-1], -10000 if metadata_missing else 0).to(attention_mask.device), attention_mask), 0) 
            
            metadata_type_row = metadata_type_embed.repeat(row_size+1, 1, 1)
            metadata_type_row[0,0,:] = pad.squeeze(0)
            
            table_stacked_embed = torch.cat((metadata_type_row, table_stacked_embed), 1)
            labels_stacked = torch.cat((torch.full(metadata_type_row.shape[:-1], -100).to(labels_stacked.device), labels_stacked), 1)
            attention_mask = torch.cat((torch.full(metadata_type_row.shape[:-1], -10000 if metadata_missing else 0).to(attention_mask.device), attention_mask), 1) 
            attention_mask[0,0] = -10000
            
            col_size += 1
            row_size += 1
            
            if col_size > max_cols:
                max_cols = col_size
            if row_size > max_rows:
                max_rows = row_size
            
            table_tensors.append(table_stacked_embed)
            attention_masks.append(attention_mask)
            labels.append(labels_stacked)
            candidates.append(candidates_encoded)

            cand_masks.append(torch.zeros((candidates_encoded.shape[0]),).to(candidates_encoded.device))

        for i, table_tensor in enumerate(table_tensors):
            
            cur_cols = table_tensor.shape[1]
            cur_rows = table_tensor.shape[0]
            cur_attention_mask = attention_masks[i]
            cur_labels = labels[i]
            cur_candidates = candidates[i]
            cur_cand_length = cur_candidates.shape[0]
            cur_cand_mask = cand_masks[i]
            
            if cur_cols < max_cols:
                col_pad = pad.repeat(cur_rows, max_cols-cur_cols, 1)
                table_tensor = torch.cat((table_tensor, col_pad), 1)
                cur_labels = torch.cat((cur_labels, torch.full(col_pad.shape[:-1], -100).to(cur_labels.device)), 1)
                cur_attention_mask = torch.cat((cur_attention_mask, torch.full(col_pad.shape[:-1], -10000).to(cur_attention_mask.device)), 1)
                
            if cur_rows < max_rows:
                row_pad = pad.repeat(max_rows-cur_rows, max_cols, 1)
                table_tensor = torch.cat((table_tensor, row_pad), 0)
                cur_labels = torch.cat((cur_labels, torch.full(row_pad.shape[:-1], -100).to(cur_labels.device)), 0)
                cur_attention_mask = torch.cat((cur_attention_mask, torch.full(row_pad.shape[:-1], -10000).to(cur_attention_mask.device)), 0)
            
            # maybe need to handle masking of padded candidates
            if max_cand > cur_cand_length:
                cand_pad = pad.repeat(max_cand-cur_cand_length, 1)
                candidates_padded = torch.cat((cur_candidates, cand_pad), 0)
                candidates[i] = candidates_padded

                cur_cand_mask = torch.cat((cur_cand_mask, torch.ones((cand_pad.shape[0],)).to(cand_pad.device)), 0)
                cand_masks[i] = cur_cand_mask

            table_tensors[i] = table_tensor

            #cur_labels = cur_labels.reshape(cur_labels.shape[0], cur_labels.shape[1], -1)
            labels[i] = cur_labels
            
            cur_attention_mask = cur_attention_mask.reshape(cur_attention_mask.shape[0], cur_attention_mask.shape[1], -1)
            attention_masks[i] = cur_attention_mask


        tables_batch = torch.stack(table_tensors)
        attention_masks_batch = torch.stack(attention_masks)
        labels_batch = torch.stack(labels)
        candidates_batch = torch.stack(candidates)
        cand_masks_batch = torch.stack(cand_masks)

        tables_batch = self.dropout(tables_batch)
        tables_batch = self.LayerNorm(tables_batch)

        candidates_batch = self.dropout(candidates_batch)
        candidates_batch = self.LayerNorm(candidates_batch)

        tables_batch = self.axial_encoder(tables_batch, attention_masks_batch)

        return tables_batch, labels_batch, candidates_batch, cand_masks_batch

class TableModelForCta(nn.Module):
    def __init__(self, frozen=False, pool=True, sum_axial=True, num_labels=-1):
        super().__init__()

        self.num_labels = num_labels
        self.pool = pool
        #config = AutoConfig.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', gradient_checkpointing=True)
        #self.cell_encoder = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', config=config)
        self.cell_encoder = AutoModelOriginal.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        
        self.type_embeddings = TypeEmbeddings(self.cell_encoder.config)
        self.pos_emb = AxialPositionalEmbedding(self.cell_encoder.config, dim = self.cell_encoder.config.hidden_size, shape = (512, 512), emb_dim_index = 3)
        
        self.LayerNorm = nn.LayerNorm(self.cell_encoder.config.hidden_size, eps=self.cell_encoder.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.cell_encoder.config.hidden_dropout_prob)
        
        axial_encoder = HTTEncoder(self.cell_encoder.config, sum_axial)
        
        self.axial_encoder = axial_encoder

        if frozen:
            for param in self.cell_encoder.parameters():
                param.requires_grad = False
        
        init_weights(self.LayerNorm)
            
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0], inputs[1])
            return inputs
        return custom_forward
    
    def forward(self, tables):
        
        max_cols = -1
        max_rows = -1
        
        table_tensors = []
        labels = []
        attention_masks = []
        
        cls_col = self.type_embeddings(None, 3)
        cls_row = self.type_embeddings(None, 4)
        cls_table = self.type_embeddings(None, 5)
        pad = self.type_embeddings(None, 7)
        
        for [[cols, cells_missing], [headers, headers_missing], [metadata, metadata_missing], [label_table]] in tables:
            
            col_tensors = []
            
            for col in cols:
                if self.pool:
                    output = self.cell_encoder(col['input_ids'], col['attention_mask'])
                    output = mean_pooling(output, col['attention_mask'])
                else:
                    output = self.cell_encoder(col['input_ids'], col['attention_mask'])['pooler_output']
                col_tensors.append(output)

            if self.pool:
                header_encoded = self.cell_encoder(headers['input_ids'], headers['attention_mask'])
                header_encoded = mean_pooling(header_encoded, headers['attention_mask'])
            else:
                header_encoded = self.cell_encoder(headers['input_ids'], headers['attention_mask'])['pooler_output']
            
            header_type_embed = self.type_embeddings(header_encoded, 1)
            
            if self.pool:
                metadata_encoded = self.cell_encoder(metadata['input_ids'], metadata['attention_mask'])
                metadata_encoded = mean_pooling(metadata_encoded, metadata['attention_mask'])
            else:
                metadata_encoded = self.cell_encoder(metadata['input_ids'], metadata['attention_mask'])['pooler_output']
            
            metadata_type_embed = self.type_embeddings(metadata_encoded, 2)

            # TODO check stacking is done correctly!
            table_stacked = torch.stack(col_tensors, dim=1)
            
            table_stacked_embed = self.type_embeddings(table_stacked, 0)   
            
            attention_mask = cells_missing

            table_stacked_embed = torch.cat((header_type_embed.unsqueeze(0), table_stacked_embed), 0)
            attention_mask = torch.cat((headers_missing.unsqueeze(0), attention_mask), 0)
            
            #TODO BUILD attention mask based on MV
            # start building attention mask to not consider MV or padding
            attention_mask[attention_mask==1] = -10000
            attention_mask[attention_mask!=-10000] = 0
            
            
            col_size = table_stacked_embed.shape[1]
            row_size = table_stacked_embed.shape[0]
            
            cls_col_embed = cls_col.repeat(1, col_size, 1)
            
            table_stacked_embed = torch.cat((cls_col_embed, table_stacked_embed), 0)
            
            attention_mask = torch.cat((torch.full(cls_col_embed.shape[:-1], 0).to(attention_mask.device), attention_mask), 0)
            
            cls_row_single = cls_row.repeat(row_size+1, 1, 1)
            cls_row_single[0,0,:] = cls_table.squeeze(0)
            
            table_stacked_embed = torch.cat((cls_row_single, table_stacked_embed), 1)
            attention_mask = torch.cat((torch.full(cls_row_single.shape[:-1], 0).to(attention_mask.device), attention_mask), 1)
            
            table_stacked_embed = self.pos_emb(table_stacked_embed)
               
            col_size += 1
            row_size += 1
            
            metadata_type_col = metadata_type_embed.repeat(1, col_size, 1)
            
            table_stacked_embed = torch.cat((metadata_type_col, table_stacked_embed), 0)
            attention_mask = torch.cat((torch.full(metadata_type_col.shape[:-1], -10000 if metadata_missing else 0).to(attention_mask.device), attention_mask), 0) 
            
            metadata_type_row = metadata_type_embed.repeat(row_size+1, 1, 1)
            metadata_type_row[0,0,:] = pad.squeeze(0)
            
            table_stacked_embed = torch.cat((metadata_type_row, table_stacked_embed), 1)
            attention_mask = torch.cat((torch.full(metadata_type_row.shape[:-1], -10000 if metadata_missing else 0).to(attention_mask.device), attention_mask), 1) 
            attention_mask[0,0] = -10000
            
            col_size += 1
            row_size += 1
            
            if col_size > max_cols:
                max_cols = col_size
            if row_size > max_rows:
                max_rows = row_size
            
            table_tensors.append(table_stacked_embed)
            attention_masks.append(attention_mask)
            labels.append(label_table)

        for i, table_tensor in enumerate(table_tensors):
            
            cur_cols = table_tensor.shape[1]
            cur_rows = table_tensor.shape[0]
            cur_attention_mask = attention_masks[i]
            cur_labels = labels[i]
            
            if cur_cols < max_cols:
                col_pad = pad.repeat(cur_rows, max_cols-cur_cols, 1)
                table_tensor = torch.cat((table_tensor, col_pad), 1)
                cur_labels = torch.cat((cur_labels, torch.full((max_cols-cur_cols, self.num_labels), -100).to(cur_labels.device)), 0)
                cur_attention_mask = torch.cat((cur_attention_mask, torch.full(col_pad.shape[:-1], -10000).to(cur_attention_mask.device)), 1)
                
            if cur_rows < max_rows:
                row_pad = pad.repeat(max_rows-cur_rows, max_cols, 1)
                table_tensor = torch.cat((table_tensor, row_pad), 0)
                cur_attention_mask = torch.cat((cur_attention_mask, torch.full(row_pad.shape[:-1], -10000).to(cur_attention_mask.device)), 0)
                
            table_tensors[i] = table_tensor

            #cur_labels = cur_labels.reshape(cur_labels.shape[0], cur_labels.shape[1], -1)
            labels[i] = cur_labels
            
            cur_attention_mask = cur_attention_mask.reshape(cur_attention_mask.shape[0], cur_attention_mask.shape[1], -1)
            attention_masks[i] = cur_attention_mask


        tables_batch = torch.stack(table_tensors)
        attention_masks_batch = torch.stack(attention_masks)
        labels_batch = torch.stack(labels)

        tables_batch = self.dropout(tables_batch)
        tables_batch = self.LayerNorm(tables_batch)

        tables_batch = self.axial_encoder(tables_batch, attention_masks_batch)

        return tables_batch, labels_batch, attention_masks_batch

        

class TypeEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding_size = config.hidden_size
        self.type_embeddings = nn.Embedding(8, config.hidden_size, padding_idx=7)

        init_weights(self.type_embeddings)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, embeds=None, embed_type=None):
        device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')
        if embeds is not None:
            type_embedding = self.type_embeddings(torch.full(embeds.shape[:len(embeds.shape)-1], embed_type).to(embeds.device))
            embeds = embeds + type_embedding
        else:
            type_embedding = self.type_embeddings(torch.full(([1]), embed_type).to(device))
            embeds = type_embedding
#         self.LayerNorm(embeds)
#         self.dropout(embeds)
        
        return embeds
    
    
class DiscriminatorHead(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dense_prediction = nn.Linear(config.hidden_size, 1)

        init_weights(self.dense)
        init_weights(self.dense_prediction)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = self.dense_prediction(hidden_states)

        return logits

class ElClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, candidates):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0],-1, shape[3])
        scores = torch.matmul(hidden_states, torch.transpose(candidates,1,2))

        return scores

class REClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, use_colcls=True):
        super().__init__()
        self.dense = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = ACT2FN[config.hidden_act]
        self.out_proj = nn.Linear(2*config.hidden_size, 121)
        self.use_colcls = use_colcls

        init_weights(self.dense)
        init_weights(self.out_proj)

    def forward(self, features, attention_masks):

        if self.use_colcls:
            #select COLCLS tokens
            cols = features[:,1,2:,:]
            obj_cols = cols[:,1:,:]
            subj_col = cols[:,0,:].unsqueeze(1).expand_as(obj_cols)
            x = torch.cat([subj_col, obj_cols], dim=-1)
            
        else:
            attention_masks = attention_masks.clone()
            attention_masks[attention_masks==0] = 1
            attention_masks[attention_masks==-10000] = 0
            attention_masks_expanded = attention_masks.expand(features.size()).float()
            features = features * attention_masks_expanded
            cols = features[:,2:,2:,:].sum(dim=1) / torch.clamp(attention_masks_expanded[:,2:,2:,:].sum(dim=1), min=1e-9)
            obj_cols = cols[:,1:,:]
            subj_col = cols[:,0,:].expand_as(obj_cols)
            x = torch.cat([subj_col, obj_cols], dim=-1)

        x = self.dropout(x)
        x = self.dense(x)
        #x = torch.tanh(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HTTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, use_colcls=True):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = ACT2FN[config.hidden_act]
        self.out_proj = nn.Linear(config.hidden_size, 255)
        self.use_colcls = use_colcls

        init_weights(self.dense)
        init_weights(self.out_proj)

    def forward(self, features, attention_masks):

        if self.use_colcls:
            #select COLCLS tokens
            x = features[:,1,2:,:]
        else:
            attention_masks = attention_masks.clone()
            attention_masks[attention_masks==0] = 1
            attention_masks[attention_masks==-10000] = 0
            attention_masks_expanded = attention_masks.expand(features.size()).float()
            features = features * attention_masks_expanded
            x = features[:,2:,2:,:].sum(dim=1) / torch.clamp(attention_masks_expanded[:,2:,2:,:].sum(dim=1), min=1e-9)

        x = self.dropout(x)
        x = self.dense(x)
        #x = torch.tanh(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class TableModelForCorruptionPretraining(nn.Module):

    def __init__(self, pos_neg_ratio=None, frozen=False, pool=True, sum_axial=True):
        super().__init__()

        self.pos_neg_ratio = pos_neg_ratio

        self.frozen = frozen
        self.pool = pool
        self.sum_axial = sum_axial

        self.table_model = TableModelForPreTraining(self.frozen, self.pool, self.sum_axial)
        #self.classifier = TableClassificationHead(self.table_model.cell_encoder.config)
        
        self.config = self.table_model.cell_encoder.config
        #self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.classifier = DiscriminatorHead(self.config)

    def forward(self, tables):
        
        outputs, labels = self.table_model(tables)
        logits = self.classifier(outputs)
        loss = None
        #loss_fct = nn.CrossEntropyLoss(reduction='sum')
        #loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.,5.]).to(logits.device), reduction='sum')
        #loss = loss_fct(logits.permute(0,3,1,2), labels.squeeze(3))
        
        if self.pos_neg_ratio is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.full([1], self.pos_neg_ratio).to(logits.device))
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        #loss_fct = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor([5]).to(logits.device))
        active_loss = (labels != -100)
        # active_logits = logits[active_loss]
        # active_labels = labels[active_loss]
        # loss = loss_fct(active_logits, active_labels.float())
        loss = loss_fct(logits, labels.float())
        active_labels = labels[active_loss].numel()
        loss = loss*active_loss.float()
        loss = torch.sum(loss)/active_labels
        
        fct = nn.Sigmoid()
        logits = torch.nan_to_num(logits, nan=-10.0)
        logits = fct(logits)

        output = (logits, labels)
        return ((loss,) + output) if loss is not None else output

#         output = (logits,) + outputs[2:]
#         return ((loss,) + output) if loss is not None else output

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, config, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]
        
        for axial_dim, axial_dim_index in zip(shape, ax_dim_indexes):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            shape = shape[1:]
            parameter = nn.Parameter(torch.randn(*shape))
            self.register_parameter(f"axial_position_{axial_dim}_{axial_dim_index}", parameter)
            init_weights(parameter)
            parameters.append(parameter)

        self.params = parameters

        #self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        #self.dropout = nn.Dropout(config.dropout)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        for param in self.params:
            if len(x.shape) == 3:
                dim_a = True if param.shape[0] == 512 else False

                if dim_a:
                    x = x + param[:x.shape[0],:,:]
                else:
                    x = x + param[:,:x.shape[1],:]
            else:
                dim_a = True if param.shape[0] == 512 else False

                if dim_a:
                    x = x + param[:x.shape[1],:,:].unsqueeze(0)
                else:
                    x = x + param[:,:x.shape[2],:].unsqueeze(0)
        
        return x
    
class HTTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        init_weights(self.dense)
        init_weights(self.LayerNorm)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class HTTAttention(nn.Module):
    def __init__(self, config, sum_axial=True):
        super().__init__()
        self.self = AxialAttention(config, dim = config.hidden_size, dim_index = 3, heads = config.num_attention_heads, num_dimensions = 2, sum_axial_out = sum_axial)
        self.output = HTTSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output
    
class HTTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        init_weights(self.dense)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class HTTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        init_weights(self.dense)
        init_weights(self.LayerNorm)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class HTTLayer(nn.Module):
    def __init__(self, config, sum_axial=True):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = HTTAttention(config, sum_axial)
        self.intermediate = HTTIntermediate(config)
        self.output = HTTOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask
        )
        attention_output = self_attention_outputs

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = layer_output

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class HTTEncoder(nn.Module):
    def __init__(self, config, sum_axial=True, gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.config = config
        self.layer = nn.ModuleList([HTTLayer(config, sum_axial) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):

        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask
                )

            hidden_states = layer_outputs
        return layer_outputs