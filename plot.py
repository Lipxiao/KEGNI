import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle 
import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Plot the loss curve')
    parser.add_argument('--file_path','-f', type=str, default='/media/disk/project/GRN/KEGNI/script/results/mHSC-E/mHSC-E_20_256_4_3_20240530_dict.pkl', help='The path of the loss file')
    return parser.parse_args()

def MultiPlot(file_path):
    plt_name = os.path.splitext(os.path.basename(file_path))[0]
    directory = os.path.dirname(file_path)

    with open(file_path, 'rb') as file:
        loss,mae_loss, kgg_kgg_loss, kgg_scg_loss, scg_scg_loss, scg_kgg_loss,epr_values,pr_values,roc_values = pickle.load(file)
    steps_range = range(0,len(mae_loss))
    pdf_path = os.path.join(directory, plt_name + '.pdf')
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize =( 8,10))
        plt.plot(steps_range, scg_kgg_loss,  label='scg_kgg_loss')
        plt.plot(steps_range, mae_loss, label='mae_loss')
        plt.plot(steps_range, kgg_scg_loss,  label='kgg_scg_loss')
        plt.plot(steps_range, kgg_kgg_loss,label='kgg_kgg_loss')

        # 调整 y 轴刻度间隔
        # plt.yticks(list(set([round(x, 1) for x in scg_kgg_loss]) | set([round(x, 1) for x in scg_kgg_loss])))
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        pdf.savefig() 
        plt.close()
        plt.figure(figsize =( 8,10))
        plt.plot(steps_range, scg_scg_loss, color='red', label='scg_scg_loss')
        plt.yticks(list(set([round(x, 1) for x in scg_scg_loss]) | set([round(x, 1) for x in scg_scg_loss])))
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        pdf.savefig() 
        plt.close()
        

        plt.figure()
        plt.plot(steps_range, mae_loss,  color='blue',label='mae_loss')
        plt.yticks(list(set([round(x, 1) for x in mae_loss]) | set([round(x, 1) for x in mae_loss])))
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        pdf.savefig() 
        plt.close()
        

        plt.figure()
        num_conditions = 3
        step = 3
        conditions_values = [epr_values[i::step] for i in range(num_conditions)]
        for i, condition_values in enumerate(conditions_values):
            plt.plot(range(len(condition_values)), condition_values)
        plt.legend("")
        plt.title('Epr')
        plt.xlabel('Steps')
        plt.ylabel('Epr')
        pdf.savefig() 
        plt.close()
    
    
        plt.figure()
        num_conditions = 3
        step = 3
        conditions_values = [pr_values[i::step] for i in range(num_conditions)]
        for i, condition_values in enumerate(conditions_values):
            plt.plot(range(len(condition_values)), condition_values)
        plt.legend("")
        plt.title('PR')
        plt.xlabel('Steps')
        plt.ylabel('PR')
        pdf.savefig() 
        plt.close()
    
        
        plt.figure()
        colors = ['red', 'green', 'blue']
        num_conditions = 3
        step = 3
        conditions_values = [roc_values[i::step] for i in range(num_conditions)]
        for i, condition_values in enumerate(conditions_values):
            plt.plot(range(len(condition_values)), condition_values)
        plt.legend("")
        plt.title('ROC')
        plt.xlabel('Steps')
        plt.ylabel('ROC')
        pdf.savefig() 
        plt.close()
    
if __name__ == '__main__':
    args = parse_args()
    MultiPlot(args.file_path) 
 

