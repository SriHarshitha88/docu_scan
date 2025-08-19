"""
ML fallback system for low-confidence document classification cases.
Uses lightweight ML models (SVM/Naive Bayes) with TF-IDF features.
"""

import pickle
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from document_classifier import DocumentType, ClassificationResult
from logger import app_logger


class MLFallbackClassifier:
    """
    ML fallback classifier for cases where heuristic confidence is below threshold.
    Uses TF-IDF features with SVM and Naive Bayes classifiers.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML fallback classifier.
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path or "models/ml_fallback_classifier.pkl"
        self.confidence_threshold = 0.70
        self.svm_pipeline = None
        self.nb_pipeline = None
        self.is_trained = False
        
        # Feature extraction parameters
        self.tfidf_params = {
            'max_features': 5000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 3),
            'stop_words': 'english',
            'lowercase': True,
            'strip_accents': 'ascii'
        }
        
        # Document type mapping for ML models
        self.type_to_label = {
            DocumentType.INVOICE: 0,
            DocumentType.RECEIPT: 1,
            DocumentType.CONTRACT: 2,
            DocumentType.BILL: 3,
            DocumentType.MEDICAL: 4,
            DocumentType.FINANCIAL: 5,
            DocumentType.LEGAL: 6,
            DocumentType.ACADEMIC: 7,
            DocumentType.GOVERNMENT: 8,
            DocumentType.UNKNOWN: 9
        }
        
        self.label_to_type = {v: k for k, v in self.type_to_label.items()}
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load pre-trained model if available."""
        model_file = Path(self.model_path)
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.svm_pipeline = model_data['svm_pipeline']
                    self.nb_pipeline = model_data['nb_pipeline']
                    self.is_trained = True
                    app_logger.info("ML fallback model loaded successfully")
                    return True
            except Exception as e:
                app_logger.warning(f"Failed to load ML model: {e}")
        
        app_logger.info("No pre-trained ML model found, will use default training data")
        self._initialize_with_synthetic_data()
        return False
    
    def _save_model(self):
        """Save trained model to file."""
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'svm_pipeline': self.svm_pipeline,
            'nb_pipeline': self.nb_pipeline
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            app_logger.info(f"ML model saved to {self.model_path}")
        except Exception as e:
            app_logger.error(f"Failed to save ML model: {e}")
    
    def _initialize_with_synthetic_data(self):
        """Initialize model with synthetic training data for cold start."""
        app_logger.info("Initializing ML fallback with synthetic training data")
        
        # Synthetic training data for each document type
        synthetic_data = {
            DocumentType.INVOICE: [
                "invoice number 12345 bill to customer name total amount due date payment terms",
                "billing statement account invoice date vendor supplier tax subtotal",
                "commercial invoice ship to bill to quantity unit price line item total"
            ],
            
            DocumentType.BILL: [
                "bill statement utility bill phone bill electricity bill water bill",
                "monthly bill billing period account number due date amount due",
                "service bill billing address previous balance current charges"
            ],
            
            DocumentType.RECEIPT: [
                "receipt purchase transaction cash total change thank you visit again",
                "store merchant cashier terminal payment method credit debit card",
                "customer copy purchase date time total tax subtotal"
            ],
            
            DocumentType.CONTRACT: [
                "agreement contract terms conditions party contractor client effective date",
                "obligation responsibility liability warranty signature witness notary",
                "term duration expiration parties agreement executed"
            ],
            
            DocumentType.MEDICAL: [
                "patient doctor physician clinic hospital diagnosis treatment medication",
                "prescription dosage mg ml tablet symptoms vital signs blood pressure",
                "medical record chart visit insurance diagnosis treatment plan"
            ],
            
            DocumentType.FINANCIAL: [
                "bank account balance statement deposit withdrawal transfer transaction",
                "interest fee charge credit routing number account number",
                "portfolio investment dividend yield financial statement"
            ],
            
            DocumentType.LEGAL: [
                "court legal lawsuit litigation plaintiff defendant attorney counsel",
                "motion filing discovery deposition statute regulation code law",
                "whereas therefore hereby jurisdiction legal document"
            ]
        }
        
        # Prepare training data
        texts = []
        labels = []
        
        for doc_type, examples in synthetic_data.items():
            for text in examples:
                texts.append(text)
                labels.append(self.type_to_label[doc_type])
        
        # Train models
        self._train_models(texts, labels)
    
    def _train_models(self, texts: List[str], labels: List[int]):
        """Train SVM and Naive Bayes models."""
        
        # Create pipelines
        self.svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**self.tfidf_params)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ])
        
        self.nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**self.tfidf_params)),
            ('nb', MultinomialNB(alpha=0.1))
        ])
        
        # Train models
        try:
            app_logger.info("Training SVM model...")
            self.svm_pipeline.fit(texts, labels)
            
            app_logger.info("Training Naive Bayes model...")
            self.nb_pipeline.fit(texts, labels)
            
            self.is_trained = True
            self._save_model()
            
            app_logger.info("ML fallback models trained successfully")
            
        except Exception as e:
            app_logger.error(f"Failed to train ML models: {e}")
            self.is_trained = False
    
    def classify_document(self, text: str) -> ClassificationResult:
        """
        Classify document using ML fallback.
        
        Args:
            text: Document text content
            
        Returns:
            ClassificationResult with ML-based classification
        """
        if not self.is_trained:
            app_logger.warning("ML fallback not trained, returning unknown classification")
            return ClassificationResult(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0
            )
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Get predictions from both models
            svm_proba = self.svm_pipeline.predict_proba([processed_text])[0]
            nb_proba = self.nb_pipeline.predict_proba([processed_text])[0]
            
            # Ensemble prediction (weighted average)
            ensemble_proba = 0.7 * svm_proba + 0.3 * nb_proba
            
            # Get best prediction
            best_label = np.argmax(ensemble_proba)
            confidence = float(ensemble_proba[best_label])
            
            # Convert back to document type
            document_type = self.label_to_type.get(best_label, DocumentType.UNKNOWN)
            
            # Build detailed features for explanation
            features = self._extract_features(processed_text)
            
            result = ClassificationResult(
                document_type=document_type,
                confidence=confidence,
                primary_indicators=features['top_features'][:3],
                secondary_indicators=features['top_features'][3:6],
                context_matches=[],
                structural_score=0.0,
                keyword_score=confidence,  # Use ML confidence as keyword score
                context_score=0.0
            )
            
            app_logger.info(f"ML fallback classification: {document_type.value} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            app_logger.error(f"ML fallback classification failed: {e}")
            return ClassificationResult(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for ML classification."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\$\%\#]', ' ', text)
        
        # Remove numbers (they can be noise for classification)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        
        # Remove very short words
        text = ' '.join(word for word in text.split() if len(word) > 2)
        
        return text.strip()
    
    def _extract_features(self, text: str) -> Dict[str, List[str]]:
        """Extract interpretable features from text for explanation."""
        try:
            # Get TF-IDF features
            tfidf_vectorizer = self.svm_pipeline.named_steps['tfidf']
            features = tfidf_vectorizer.transform([text])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Get top features by TF-IDF score
            feature_scores = features.toarray()[0]
            top_indices = np.argsort(feature_scores)[-10:][::-1]
            
            top_features = [feature_names[i] for i in top_indices if feature_scores[i] > 0]
            
            return {
                'top_features': top_features,
                'feature_scores': [float(feature_scores[i]) for i in top_indices]
            }
            
        except Exception as e:
            app_logger.warning(f"Feature extraction failed: {e}")
            return {'top_features': [], 'feature_scores': []}
    
    def update_model(self, texts: List[str], labels: List[int]):
        """
        Update model with new training data.
        
        Args:
            texts: List of document texts
            labels: List of corresponding labels (as integers)
        """
        if not texts or not labels:
            app_logger.warning("No training data provided for model update")
            return
        
        app_logger.info(f"Updating ML model with {len(texts)} new samples")
        
        try:
            # Retrain models with new data
            self._train_models(texts, labels)
            app_logger.info("Model updated successfully")
            
        except Exception as e:
            app_logger.error(f"Model update failed: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        return {
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'supported_types': list(self.type_to_label.keys()),
            'feature_params': self.tfidf_params
        }


class HybridClassifier:
    """
    Hybrid classifier that uses heuristics first, then ML fallback for low confidence.
    """
    
    def __init__(self, heuristic_classifier, ml_fallback: Optional[MLFallbackClassifier] = None):
        """
        Initialize hybrid classifier.
        
        Args:
            heuristic_classifier: Primary heuristic classifier
            ml_fallback: ML fallback classifier (optional)
        """
        self.heuristic_classifier = heuristic_classifier
        self.ml_fallback = ml_fallback or MLFallbackClassifier()
        self.confidence_threshold = 0.70
    
    def classify_document(self, text: str) -> ClassificationResult:
        """
        Classify document using hybrid approach.
        
        Args:
            text: Document text content
            
        Returns:
            ClassificationResult from best available method
        """
        app_logger.info("Starting hybrid document classification")
        
        # Stage 1: Try heuristic classification
        heuristic_result = self.heuristic_classifier.classify_document(text)
        
        app_logger.info(f"Heuristic result: {heuristic_result.document_type.value} "
                       f"(confidence: {heuristic_result.confidence:.2f})")
        
        # If confidence is high enough, use heuristic result
        if heuristic_result.confidence >= self.confidence_threshold:
            app_logger.info("Using heuristic classification (high confidence)")
            return heuristic_result
        
        # Stage 2: Use ML fallback for low confidence cases
        app_logger.info("Using ML fallback (low heuristic confidence)")
        ml_result = self.ml_fallback.classify_document(text)
        
        app_logger.info(f"ML result: {ml_result.document_type.value} "
                       f"(confidence: {ml_result.confidence:.2f})")
        
        # Choose best result based on confidence
        if ml_result.confidence > heuristic_result.confidence:
            # Enhance ML result with heuristic insights
            ml_result.primary_indicators.extend(heuristic_result.primary_indicators[:2])
            ml_result.context_matches = heuristic_result.context_matches
            return ml_result
        else:
            return heuristic_result
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for ML fallback trigger."""
        self.confidence_threshold = threshold
        app_logger.info(f"Confidence threshold set to {threshold}")


def create_hybrid_classifier(heuristic_classifier) -> HybridClassifier:
    """Factory function to create a hybrid classifier."""
    return HybridClassifier(heuristic_classifier)