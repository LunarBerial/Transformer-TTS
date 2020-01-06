from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():

    dataset = get_dataset()
    global_step = 0
    # inference： https://blog.csdn.net/weixin_40087578/article/details/87186613
    m = nn.DataParallel(Model().cuda()) # 将data分配给多GPU，默认用0号卡训练。如使用多卡，需提前指定device编号并设置环境变量

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr) # Adam

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter()
    
    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step) # 调整学习率。但对Adam来说，似乎没什么必要。
            # pos_text和pos_mel是全局排序。
            character, mel, mel_input, pos_text, pos_mel, _ = data #取data
            
            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
            
            character = character.cuda() #data拷贝至GPU
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)
            # 这里的stop_token原本是用来标记音频结尾的符号。但代码作者表示，按原文加上loss会使模型不收敛。后续生成的 时候也只能凭借经验值确定生成长度。
            mel_loss = nn.L1Loss()(mel_pred, mel) # L1 loss
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)
            
            loss = mel_loss + post_mel_loss
            
            writer.add_scalars('training_loss',{
                    'mel_loss':mel_loss,
                    'post_mel_loss':post_mel_loss,

                }, global_step)
                
            writer.add_scalars('alphas',{
                    'encoder_alpha':m.module.encoder.alpha.data,
                    'decoder_alpha':m.module.decoder.alpha.data,
                }, global_step)
            
            
            if global_step % hp.image_step == 1:
                
                for i, prob in enumerate(attn_probs):
                    
                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_%d_0'%global_step, x, i*4+j)
                
                for i, prob in enumerate(attns_enc):
                    num_h = prob.size(0)
                    
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
            
                for i, prob in enumerate(attns_dec):

                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)
                
            optimizer.zero_grad() # 手动清零梯度数组，方便下次计算。
            # Calculate gradients
            loss.backward() # BP
            
            nn.utils.clip_grad_norm_(m.parameters(), 1.) # 梯度裁剪
            
            # Update weights 更新权重。
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))

            
            


if __name__ == '__main__':
    main()