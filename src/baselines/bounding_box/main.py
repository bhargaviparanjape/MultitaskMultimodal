__author__ = 'aditichaudhary'

from utilities import *
from dataloader import DataLoader
import random
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

IS_CUDA = torch.cuda.is_available()

def batchify(datapoints, batch_size, time_steps):
    datapoints = [token for answer in datapoints for token in answer]
    datapoints = np.array(datapoints, dtype=np.int32)
    data_len = len(datapoints)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = datapoints[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // time_steps

    for i in range(epoch_size):
        x = data[:, i * time_steps:(i + 1) * time_steps]
        y = data[:, i * time_steps + 1 :(i + 1) * time_steps  + 1 ]
        yield (x, y)

def eval(model, datapoints):

    tot_examples = 0
    cumulative_loss = 0
    start_time = time.time()

    for step_num, (input, target) in enumerate(
            batchify(datapoints, args.batch_size, args.bptt)):  # (batch_size, time_steps)


        input = np.array(input, dtype=int)
        input_variable = torch.autograd.Variable(LongTensor(input).transpose(0, 1).contiguous())

        target = np.array(target, dtype=int)
        target_variable = torch.autograd.Variable(LongTensor(target).transpose(0, 1).contiguous())

        input_seq_length = [len(b) for b in input]
        loss = model(input_variable, input_seq_length, target_variable)
        cumulative_loss += loss.data.item()
        tot_examples += 1

    print("Evaluation: perplexity: {:8.2f} speed: {} wps".format(
                                                                       np.exp(cumulative_loss / tot_examples),
                                                                       tot_examples * args.batch_size / (
                                                                       time.time() - start_time)))
    #model.eval(False)


def train(model, train_datapoints, valid_datapoints):



    validation_history = []
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    validation_history.append(0)

    epoch_size = (len(train_datapoints) -1) // args.bptt
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        tot_examples = 0
        cumulative_loss = 0
        np.random.shuffle(train_datapoints)
        for step_num, (input, target) in enumerate(batchify(train_datapoints,args.batch_size, args.bptt)): #(batch_size, time_steps)
            optimizer.zero_grad()

            input = np.array(input, dtype=int)
            input_variable = torch.autograd.Variable(LongTensor(input).transpose(0,1).contiguous())

            target = np.array(target, dtype=int)
            target_variable = torch.autograd.Variable(LongTensor(target).transpose(0, 1).contiguous())

            input_seq_length = [len(b) for b in input]
            loss = model(input_variable, input_seq_length, target_variable)
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.data.item()
            tot_examples += 1

            if step_num % epoch_size  == 0:

                print("Epoch: {}  Itr:{} perplexity: {:8.2f} speed: {} wps".format(epoch, step_num, np.exp(cumulative_loss / tot_examples),
                                                                    tot_examples * args.batch_size / (time.time() - start_time)))
                eval(model, valid_datapoints)
                model.train()


def writeOutput(ids, file):
    print("Writing to file {0}".format(file))
    fout = codecs.open(file, "w", encoding='utf-8')
    for id in ids:
        fout.write(str(id) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Data
    parser.add_argument('--train_question_path',type=str, default="/Users/aditichaudhary/Documents/CMU/11777/v2_OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument('--valid_question_path', type=str, default="/Users/aditichaudhary/Documents/CMU/11777/v2_OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument('--test_question_path', type=str, default="/Users/aditichaudhary/Documents/CMU/11777/v2_OpenEnded_mscoco_test2015_questions.json")
    parser.add_argument('--dictionary_path', type=str, default=None)
    parser.add_argument('--train_answer_path', type=str, default="/Users/aditichaudhary/Documents/CMU/11777/train_target.pkl")
    parser.add_argument('--valid_answer_path', type=str, default="/Users/aditichaudhary/Documents/CMU/11777/val_target.pkl")
    parser.add_argument('--label2answer_path', type=str, default="/Users/aditichaudhary/Documents/CMU/11777/trainval_label2ans.pkl")

    parser.add_argument("--train_expressions", type=str)
    parser.add_argument("--valid_expressions", type=str)
    # Model arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=3)
    parser.add_argument('--model_path', type=str, default='saved_models/default.model')
    parser.add_argument('--word_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument("--debug_text", type=str, default='/Users/aditichaudhary/Documents/CMU/text8')
    parser.add_argument("--imageids", type=str, default=None)
    parser.add_argument("--mode", default="question", type=str,
                        choices=["question", "answer", "caption","Ref-VQA"],
                        help="questions of vqa, answers of vqa, captions of GoogleRef")
    parser.add_argument("--rnncell", default="GRU", type=str,
                        choices=["GRU", "LSTM"],
                        help="gru, lstm")

    args = parser.parse_args()
    print(args)

    dataloader = DataLoader(args)

    print("Loading VQA:{0}".format(args.mode))
    train_vqa_ids = dataloader.load_dataset(args.train_question_path)
    writeOutput(train_vqa_ids, args.imageids + "/vqa_train.ids")

    valid_vqa_ids = dataloader.load_dataset(args.valid_question_path)
    writeOutput(train_vqa_ids, args.imageids + "/vqa_valid.ids")

    test_vqa_ids = dataloader.load_dataset(args.test_question_path)
    writeOutput(train_vqa_ids, args.imageids + "/vqa_test.ids")

    print("Train VQA :{0}, Valid VQA :{1} Test VQA:{2}".format(len(train_vqa_ids), len(valid_vqa_ids), len(test_vqa_ids)))
	

    print("Loading Googleref expressions")
    train_ref_ids = dataloader.load_captions(args.train_expressions,analysze=False, init=True)
    writeOutput(train_vqa_ids, args.imageids + "/ref_train.ids")

    valid_ref_ids = dataloader.load_captions(args.valid_expressions, analysze=False, init=True)
    writeOutput(train_vqa_ids, args.imageids + "/vqa_valid.ids")

    print("Train Ref :{0} Valid Ref {1}" .format(len(train_ref_ids), len(valid_ref_ids)))


    intersection = list(set(train_vqa_ids) & set(valid_vqa_ids) & set(test_vqa_ids) & set(train_ref_ids) & set(valid_ref_ids))
    print("Intersection :{0}".format(len(intersection)))
    writeOutput(intersection, args.imageids + "/intersection.ids")


    # intersection = list(set(vqa_ids) & set(ref_ids))
    # random.shuffle(intersection)
    # print("Intersection :{0}".format(len(intersection)))

    #fout= codecs.open(args.imageids,"w", encoding='utf-8')
    #intersection = intersection[:1000]


