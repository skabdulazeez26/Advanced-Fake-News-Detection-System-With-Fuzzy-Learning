#!/usr/bin/env python3
"""
Training script for LIAR2 dataset (5-module FDHN)
Following research paper specifications
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import sys
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Add src to path
sys.path.append('src')

from models.fdhn_liar2 import FDHN_LIAR2
from data.data_processor import DataProcessor, LIAR2Dataset

def train_liar2_model():
    """Train FDHN on LIAR2 dataset"""
    
    print("=" * 60)
    print("Training FDHN on LIAR2 Dataset (5-module)")
    print("=" * 60)
    
    # Initialize processor
    processor = DataProcessor()
    
    # Load LIAR2 data
    train_df, valid_df, test_df = processor.load_liar2_data(
        'data/liar2/train.csv',
        'data/liar2/valid.csv',
        'data/liar2/test.csv'
    )
    
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Build vocabularies
    all_statements = train_df['statement'].fillna('').tolist()
    all_contexts = train_df[['subject', 'speaker', 'speaker_description', 'state_info', 'context']].fillna('').apply(
        lambda x: ' '.join(x.astype(str)), axis=1
    ).tolist()
    all_justifications = train_df['justification'].fillna('').tolist()
    
    statement_vocab = processor.build_vocabulary(all_statements)
    context_vocab = processor.build_vocabulary(all_contexts)
    justification_vocab = processor.build_vocabulary(all_justifications)
    
    print(f"Statement vocab: {len(statement_vocab)}")
    print(f"Context vocab: {len(context_vocab)}")
    print(f"Justification vocab: {len(justification_vocab)}")
    
    # Process data
    def process_split(df):
        statements = df['statement'].fillna('').tolist()
        contexts = df[['subject', 'speaker', 'speaker_description', 'state_info', 'context']].fillna('').apply(
            lambda x: ' '.join(x.astype(str)), axis=1
        ).tolist()
        justifications = df['justification'].fillna('').tolist()
        
        # Numerical features (6 credibility counts including true_counts)
        numerical = df[['true_counts', 'mostly_true_counts', 'half_true_counts', 
                       'mostly_false_counts', 'false_counts', 'pants_on_fire_counts']].fillna(0).values
        
        # Scale numerical features
        if hasattr(processor.scaler, 'scale_'):
            numerical = processor.scaler.transform(numerical)
        else:
            numerical = processor.scaler.fit_transform(numerical)
        
        # Encode labels
        labels = df['label'].tolist()
        if hasattr(processor.label_encoder, 'classes_'):
            encoded_labels = processor.label_encoder.transform(labels)
        else:
            encoded_labels = processor.label_encoder.fit_transform(labels)
        
        return statements, contexts, justifications, numerical, encoded_labels
    
    # Process all splits
    train_statements, train_contexts, train_justifications, train_numerical, train_labels = process_split(train_df)
    valid_statements, valid_contexts, valid_justifications, valid_numerical, valid_labels = process_split(valid_df)
    test_statements, test_contexts, test_justifications, test_numerical, test_labels = process_split(test_df)
    
    # Create datasets
    train_dataset = LIAR2Dataset(train_statements, train_contexts, train_justifications, train_numerical, train_labels,
                                statement_vocab, context_vocab, justification_vocab)
    valid_dataset = LIAR2Dataset(valid_statements, valid_contexts, valid_justifications, valid_numerical, valid_labels,
                                statement_vocab, context_vocab, justification_vocab)
    test_dataset = LIAR2Dataset(test_statements, test_contexts, test_justifications, test_numerical, test_labels,
                               statement_vocab, context_vocab, justification_vocab)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = FDHN_LIAR2(
        vocab_size_news=len(statement_vocab),
        vocab_size_context=len(context_vocab),
        vocab_size_justification=len(justification_vocab),
        numerical_dim=6
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_valid_acc = 0
    epochs = 10
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            statements = batch['statement'].to(device)
            contexts = batch['context'].to(device)
            justifications = batch['justification'].to(device)
            numerical = batch['numerical'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(statements, contexts, justifications, numerical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_true.extend(labels.cpu().numpy())
        
        # Validate
        model.eval()
        valid_loss = 0
        valid_preds, valid_true = [], []
        
        with torch.no_grad():
            for batch in valid_loader:
                statements = batch['statement'].to(device)
                contexts = batch['context'].to(device)
                justifications = batch['justification'].to(device)
                numerical = batch['numerical'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(statements, contexts, justifications, numerical)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                valid_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                valid_true.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(train_true, train_preds)
        valid_acc = accuracy_score(valid_true, valid_preds)
        valid_f1 = f1_score(valid_true, valid_preds, average='macro')
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}")
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'valid_acc': valid_acc,
                'valid_f1': valid_f1
            }, 'models/liar2_model.pth')
    
    # Test evaluation
    model.eval()
    test_preds, test_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            statements = batch['statement'].to(device)
            contexts = batch['context'].to(device)
            justifications = batch['justification'].to(device)
            numerical = batch['numerical'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(statements, contexts, justifications, numerical)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average='macro')
    
    print("=" * 60)
    print("LIAR2 Model Results:")
    print(f"Best Valid Acc: {best_valid_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print("=" * 60)
    
    # Save processor
    processor.statement_vocab = statement_vocab
    processor.context_vocab = context_vocab
    processor.justification_vocab = justification_vocab
    
    # Create processor data without lambda
    processor_data = {
        'statement_vocab': statement_vocab,
        'context_vocab': context_vocab,
        'justification_vocab': justification_vocab,
        'label_encoder': processor.label_encoder,
        'scaler': processor.scaler,
        'model_params': {
            'vocab_size_news': len(statement_vocab),
            'vocab_size_context': len(context_vocab),
            'vocab_size_justification': len(justification_vocab),
            'numerical_dim': 6
        }
    }
    
    with open('models/liar2_processor.pkl', 'wb') as f:
        pickle.dump(processor_data, f)
    
    print("✅ LIAR2 model and processor saved!")

if __name__ == "__main__":
    train_liar2_model()