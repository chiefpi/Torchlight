"""Trinary XOR operation"""
import dynet as dy


def create_xor_instances(epochs=2000):
    questions = []
    answers = []
    for _ in range(epochs):
        for x1 in 0,1:
            for x2 in 0,1:
                for x3 in 0,1:
                    answer = 1 if (x1+x2+x3)%2 else 0
                    questions.append((x1,x2,x3))
                    answers.append(answer)

    return questions, answers


def create_xor_network(W, V, b, question, answer):
    dy.renew_cg() # new computation graph
    x = dy.vecInput(len(question))
    x.set(question)
    y = dy.scalarInput(answer)
    output = dy.logistic(V*(dy.tanh((W*x)+b)))
    loss =  dy.binary_log_loss(output, y)

    return output, loss


def train(W, V, b, questions, answers, trainer):
    total_loss = 0
    step = 0
    print('=====Training=====')
    for question, answer in zip(questions, answers):
        _, loss = create_xor_network(W, V, b, question, answer) # dynamic
        step += 1
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        if step % 100 == 0:
            print("step: %4d, average loss: %.5f" % (step, total_loss/step))


def test(W, V, b, questions, answers):
    print('=====Testing=====')
    for question, answer in zip (questions, answers):
        output, _ = create_xor_network(W, V, b, question, answer) # is it ok
        print(question, "ground_truth: %d" % answer, "predicted: %.5f" % output.value())


def main():
    m = dy.ParameterCollection()
    W = m.add_parameters((8,3))
    V = m.add_parameters((1,8))
    b = m.add_parameters((8))

    trainer = dy.SimpleSGDTrainer(m)
    q_train, ans_train = create_xor_instances()
    train(W, V, b, q_train, ans_train, trainer)
    # m.save('saved_models/xor.model')

    q_test, ans_test = create_xor_instances(1)
    test(W, V, b, q_test, ans_test)


if __name__ == "__main__":
    main()