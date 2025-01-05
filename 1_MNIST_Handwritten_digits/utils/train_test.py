import torch


def training(model, train_loader, optim_fn, loss_fn):
    #code inspired from the one provided in previous lessons/exercises in "Introduction to Deep Learning"
    
    n_epochs = 10
    train_loss_h = []
    
    #training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
                
            optim_fn.zero_grad()
            out = model(inputs)
            loss = loss_fn(out, labels)
            loss.backward()
            optim_fn.step()
            _, prediction = torch.max(out.data, 1)
            train_loss += loss.item()
            train_correct += ((prediction == labels)*100.0).mean().item()
            tr_accuracy = train_correct/len(train_loader)
            tr_loss = train_loss/len(train_loader)
        print(f'Epoch {epoch + 1} training accuracy:{tr_accuracy:.2f}% training loss:{tr_loss:.5f}')
        train_loss_h.append(tr_loss)
    return train_loss_h

def testing(model, test_loader, loss_fn):
    #testing
    model.eval()
    test_loss_h = []
    test_loss = 0.0
    test_correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
            
        #testing doesn't need optimization function
        out = model(inputs)
        loss = loss_fn(out, labels)
            
        _, prediction = torch.max(out.data, 1)
        test_loss += loss.item()
        test_correct += ((prediction == labels)*100.0).mean().item()
        ts_accuracy = test_correct/len(test_loader)
        ts_loss = test_loss/len(test_loader)
    test_loss_h.append(ts_loss)
    print(f'test accuracy:{ts_accuracy:.2f}% test loss:{ts_loss:.5f}')