"""
Feature Engineering Module for Resume Skill Recognition System
Generates embeddings and features for ML models.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:  # optional ONNX runtime for faster inference
    import onnxruntime as ort
except Exception:  # noqa: BLE001
    ort = None

try:  # optional SHAP for explanations
    import shap
except Exception:  # noqa: BLE001
    shap = None

from utils import get_logger, config, cosine_similarity, safe_divide


logger = get_logger(__name__)


class FeatureEngineer:
    """Generates embeddings and features for text and skills."""
    
    def __init__(self):
        """Initialize feature engineer with embedding model."""
        self.model_name = config.get('feature_engineering.embedding_model',
                                     'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = config.get('feature_engineering.embedding_dim', 384)
        self.cache_embeddings = config.get('feature_engineering.cache_embeddings', True)
        self.normalize_vectors = config.get('feature_engineering.normalize_vectors', True)
        self.batch_size = config.get('feature_engineering.batch_size', 32)
        self.meta_model_path = config.get('matching.meta_model_path', None)
        self.meta_model_trained = config.get('matching.meta_model_trained', False)
        self.meta_model = None
        self.fuzzy_weight = config.get('matching.fuzzy_weight', 0.15)
        self.graph_weight = config.get('matching.graph_weight', 0.1)
        self.semantic_weight = config.get('matching.semantic_weight', 0.5)
        self.jaccard_weight = config.get('matching.jaccard_weight', 0.25)
        self.context_weight = config.get('matching.context_weight', 0.1)
        self.domain_weight = config.get('matching.domain_weight', 0.1)
        self.use_onnx = config.get('feature_engineering.use_onnx', False)
        self.onnx_model_path = config.get('feature_engineering.onnx_model_path', None)
        self.enable_contextual_signals = config.get('feature_engineering.enable_contextual_signals', False)
        self.roberta_model_name = config.get('feature_engineering.roberta_model_name', 'roberta-base')
        self.deberta_class_model = config.get('feature_engineering.deberta_class_model', 'microsoft/deberta-v3-base')
        self.roberta_domain_model = config.get('feature_engineering.roberta_domain_model', 'roberta-base')
        
        # Initialize embedding model
        self.model = None
        self._initialize_model()

        # Optional meta learner
        self._initialize_meta_model()
        
        # Embedding cache
        self.embedding_cache = {}
        
        logger.info("FeatureEngineer initialized with model: %s", self.model_name)
        if not self.meta_model_trained:
            logger.info("Meta-learner not provided or not fine-tuned; using weighted fusion fallback")
    
    def _initialize_model(self):
        """Initialize sentence transformer model."""
        if self.use_onnx and self.onnx_model_path and ort is not None:
            self._initialize_onnx_session()
            return

        try:
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error("Failed to load embedding model: %s", str(e))
            raise

    def _initialize_onnx_session(self):
        """Load ONNX Runtime session for embeddings when configured."""
        if ort is None:
            logger.warning("onnxruntime not installed; falling back to SentenceTransformer")
            self.use_onnx = False
            return
        model_path = Path(self.onnx_model_path)
        if not model_path.exists():
            logger.warning("ONNX model path does not exist: %s", model_path)
            self.use_onnx = False
            return
        try:
            logger.info("Loading ONNX embedding model from %s", model_path)
            self.ort_session = ort.InferenceSession(str(model_path))
            self.ort_input = self.ort_session.get_inputs()[0].name
            self.ort_output = self.ort_session.get_outputs()[0].name
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load ONNX model: %s", exc)
            self.use_onnx = False

    def _initialize_meta_model(self):
        """Load optional XGBoost meta-learner if available; otherwise remain in fallback mode."""
        if not self.meta_model_path:
            logger.debug("No meta-learner path configured; using weighted fusion")
            return

        try:  # XGBoost optional
            import xgboost as xgb
            model_path = Path(self.meta_model_path)
            if model_path.exists():
                self.meta_model = xgb.XGBRegressor()
                self.meta_model.load_model(str(model_path))
                logger.info("Loaded meta-learner from %s", model_path)
            else:
                logger.warning("Meta-learner path not found: %s", model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Meta-learner unavailable: %s", exc)
            self.meta_model = None
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        # Check cache
        if use_cache and self.cache_embeddings and text in self.embedding_cache:
            logger.debug("Using cached embedding")
            return self.embedding_cache[text]
        
        # Generate embedding
        try:
            if self.use_onnx and hasattr(self, 'ort_session'):
                embedding = self._encode_with_onnx(text)
            else:
                embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Normalize if configured
            if self.normalize_vectors:
                embedding = self._normalize_vector(embedding)
            
            # Cache embedding
            if self.cache_embeddings:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error("Failed to generate embedding: %s", str(e))
            return np.zeros(self.embedding_dim)
    
    def generate_embeddings_batch(self, texts: List[str],
                                   show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.zeros((0, self.embedding_dim))
        
        logger.info("Generating embeddings for batch of %d texts", len(texts))
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Normalize if configured
            if self.normalize_vectors:
                embeddings = np.array([self._normalize_vector(emb) for emb in embeddings])
            
            return embeddings
            
        except Exception as e:
            logger.error("Failed to generate batch embeddings: %s", str(e))
            return np.zeros((len(texts), self.embedding_dim))
    
    def generate_skill_embedding(self, skills: Dict[str, List[str]]) -> np.ndarray:
        """
        Generate embedding for skills dictionary.
        
        Args:
            skills: Dictionary with categorized skills
            
        Returns:
            Embedding vector
        """
        # Flatten all skills into a single text
        all_skills = []
        for skill_list in skills.values():
            all_skills.extend(skill_list)
        
        if not all_skills:
            return np.zeros(self.embedding_dim)
        
        # Create text representation
        skill_text = ', '.join(all_skills)
        
        return self.generate_embedding(skill_text)
    
    def generate_weighted_skill_embedding(self, skills: Dict[str, List[str]],
                                         weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Generate weighted embedding for skills with category weights.
        
        Args:
            skills: Dictionary with categorized skills
            weights: Category weights (defaults to config)
            
        Returns:
            Weighted embedding vector
        """
        if weights is None:
            weights = {
                'technical_skills': config.get('matching.weights.technical_skills', 0.5),
                'tools': config.get('matching.weights.tools', 0.3),
                'frameworks': config.get('matching.weights.frameworks', 0.15),
                'soft_skills': config.get('matching.weights.soft_skills', 0.05)
            }
        
        # Generate embeddings for each category
        category_embeddings = []
        category_weights = []
        
        for category, skill_list in skills.items():
            if skill_list and category in weights:
                skill_text = ', '.join(skill_list)
                embedding = self.generate_embedding(skill_text)
                category_embeddings.append(embedding)
                category_weights.append(weights[category])
        
        if not category_embeddings:
            return np.zeros(self.embedding_dim)
        
        # Compute weighted average
        category_embeddings = np.array(category_embeddings)
        category_weights = np.array(category_weights)
        
        # Normalize weights
        category_weights = category_weights / np.sum(category_weights)
        
        # Weighted sum
        weighted_embedding = np.sum(
            category_embeddings * category_weights[:, np.newaxis],
            axis=0
        )
        
        # Normalize result
        if self.normalize_vectors:
            weighted_embedding = self._normalize_vector(weighted_embedding)
        
        return weighted_embedding
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        return cosine_similarity(embedding1, embedding2)

    def compute_quadruple_features(self, resume_text: str, jd_text: str,
                                   resume_skills: Dict[str, List[str]],
                                   jd_skills: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute semantic, Jaccard, fuzzy, graph, and optional domain/context scores."""
        semantic_score = self.compute_similarity(
            self.generate_embedding(resume_text),
            self.generate_embedding(jd_text)
        )

        resume_flat = {s.lower() for lst in resume_skills.values() for s in lst}
        jd_flat = {s.lower() for lst in jd_skills.values() for s in lst}
        intersection = resume_flat & jd_flat
        union = resume_flat | jd_flat
        jaccard_score = safe_divide(len(intersection), len(union), default=0.0)

        fuzzy_score = self._compute_fuzzy_score(resume_text, jd_text)
        graph_score = self._compute_graph_score(intersection, jd_flat, resume_flat)

        features = {
            'semantic': semantic_score,
            'jaccard': jaccard_score,
            'fuzzy': fuzzy_score,
            'graph': graph_score
        }

        # Optional contextual/classifier signals
        contextual = self.generate_contextual_signals(resume_text, jd_text)
        features.update(contextual)

        # Optional domain relevance using RoBERTa classifier
        domain_relevance = self._compute_domain_relevance(resume_text, jd_text)
        if domain_relevance is not None:
            features['domain_relevance'] = domain_relevance

        return features

    def _encode_with_onnx(self, text: str) -> np.ndarray:
        """Encode text using ONNX Runtime session."""
        if not hasattr(self, 'ort_session'):
            return np.zeros(self.embedding_dim)
        # Simple single-sentence encode; assumes model expects input_ids already tokenized.
        try:
            # Fallback: use SentenceTransformer tokenizer to produce inputs compatible with ONNX export
            tokenizer = SentenceTransformer(self.model_name).tokenizer
            inputs = tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=256)
            ort_out = self.ort_session.run([self.ort_output], {self.ort_input: inputs['input_ids']})[0]
            # Assume first vector is pooled CLS
            return np.array(ort_out[0])
        except Exception as exc:  # noqa: BLE001
            logger.error("ONNX encoding failed: %s", exc)
            return np.zeros(self.embedding_dim)

    def hybrid_similarity(self, features: Dict[str, float]) -> float:
        """Fuse quadruple features using meta-learner or weighted sum."""
        if self.meta_model is not None:
            ordered = np.array([
                features.get('semantic', 0.0),
                features.get('jaccard', 0.0),
                features.get('fuzzy', 0.0),
                features.get('graph', 0.0),
                features.get('context_match', 0.0),
                features.get('domain_relevance', 0.0)
            ]).reshape(1, -1)
            try:
                return float(self.meta_model.predict(ordered)[0])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Meta-learner prediction failed, falling back: %s", exc)

        # Weighted sum fallback
        total_weight = (
            self.semantic_weight + self.jaccard_weight + self.fuzzy_weight +
            self.graph_weight + self.context_weight + self.domain_weight
        )
        if total_weight == 0:
            return 0.0

        return float(
            (features.get('semantic', 0.0) * self.semantic_weight +
             features.get('jaccard', 0.0) * self.jaccard_weight +
             features.get('fuzzy', 0.0) * self.fuzzy_weight +
             features.get('graph', 0.0) * self.graph_weight +
             features.get('context_match', 0.0) * self.context_weight +
             features.get('domain_relevance', 0.0) * self.domain_weight) / total_weight
        )

    def explain_with_shap(self, features: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Compute SHAP values for meta-learner when available."""
        if self.meta_model is None or shap is None:
            return None
        try:
            ordered = np.array([
                features.get('semantic', 0.0),
                features.get('jaccard', 0.0),
                features.get('fuzzy', 0.0),
                features.get('graph', 0.0),
                features.get('context_match', 0.0),
                features.get('domain_relevance', 0.0)
            ]).reshape(1, -1)
            explainer = shap.TreeExplainer(self.meta_model)
            shap_values = explainer.shap_values(ordered)
            # shap_values can be list for multiclass; handle first element
            vals = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]
            return {
                'semantic': float(vals[0]),
                'jaccard': float(vals[1]),
                'fuzzy': float(vals[2]),
                'graph': float(vals[3]),
                'context_match': float(vals[4]),
                'domain_relevance': float(vals[5])
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("SHAP explanation failed: %s", exc)
            return None

    def generate_contextual_signals(self, resume_text: str, jd_text: str) -> Dict[str, float]:
        """Placeholder contextual signals from DeBERTa/RoBERTa classifiers if configured."""
        if not self.enable_contextual_signals:
            return {}
        signals: Dict[str, float] = {}
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            logger.debug("Transformers classification models unavailable: %s", exc)
            return signals

        try:
            # Lazy load classification model
            if not hasattr(self, 'context_model'):
                self.context_tokenizer = AutoTokenizer.from_pretrained(self.deberta_class_model)
                self.context_model = AutoModelForSequenceClassification.from_pretrained(self.deberta_class_model)
            inputs = self.context_tokenizer(resume_text, jd_text, return_tensors='pt', truncation=True, padding=True, max_length=256)
            outputs = self.context_model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            signals['context_match'] = float(probs[0].max().item())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Contextual signal computation failed: %s", exc)

        return signals

    def _compute_domain_relevance(self, resume_text: str, jd_text: str) -> Optional[float]:
        """Compute domain relevance using a RoBERTa classifier trained on resume-JD pairs."""
        if not self.enable_contextual_signals:
            return None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            logger.debug("Transformers classification models unavailable: %s", exc)
            return None

        try:
            if not hasattr(self, 'domain_model'):
                self.domain_tokenizer = AutoTokenizer.from_pretrained(self.roberta_domain_model)
                self.domain_model = AutoModelForSequenceClassification.from_pretrained(self.roberta_domain_model)
            inputs = self.domain_tokenizer(resume_text, jd_text, return_tensors='pt', truncation=True, padding=True, max_length=256)
            outputs = self.domain_model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            return float(probs[0].max().item())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Domain relevance computation failed: %s", exc)
            return None

    def _compute_fuzzy_score(self, resume_text: str, jd_text: str) -> float:
        """Compute fuzzy token sort ratio using RapidFuzz if available."""
        try:  # rapidfuzz optional
            from rapidfuzz import fuzz
            return fuzz.token_sort_ratio(resume_text, jd_text) / 100.0
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fuzzy scoring unavailable: %s", exc)
            return 0.0

    def _compute_graph_score(self, intersection: set, jd_flat: set, resume_flat: Optional[set] = None) -> float:
        """Lightweight graph-like relevance using coverage + diversity heuristics (no Node2Vec)."""
        if not jd_flat:
            return 0.0

        coverage = safe_divide(len(intersection), len(jd_flat))

        # Diversity bonus: how many unique JD skills are at least partially matched vs resume size
        diversity = 0.0
        if resume_flat:
            diversity = safe_divide(len(intersection), max(len(resume_flat), 1))

        # Simple co-occurrence proxy: favor matches that hit multiple JD skills
        co_occurrence = 1.0 if len(intersection) >= 3 else safe_divide(len(intersection), 3)

        # Weighted combination to mimic a graph score without a KG
        return float(0.6 * coverage + 0.25 * diversity + 0.15 * co_occurrence)
    
    def save_cache(self, file_path: str):
        """
        Save embedding cache to file.
        
        Args:
            file_path: Path to save cache
        """
        if not self.embedding_cache:
            logger.warning("No embeddings in cache to save")
            return
        
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            
            logger.info("Saved %d cached embeddings to %s",
                       len(self.embedding_cache), file_path)
        except Exception as e:
            logger.error("Failed to save embedding cache: %s", str(e))
    
    def load_cache(self, file_path: str):
        """
        Load embedding cache from file.
        
        Args:
            file_path: Path to cache file
        """
        try:
            with open(file_path, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            
            logger.info("Loaded %d cached embeddings from %s",
                       len(self.embedding_cache), file_path)
        except Exception as e:
            logger.error("Failed to load embedding cache: %s", str(e))
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        logger.info("Cleared embedding cache")
    
    def get_cache_size(self) -> int:
        """
        Get number of cached embeddings.
        
        Returns:
            Cache size
        """
        return len(self.embedding_cache)


def generate_embedding(text: str) -> np.ndarray:
    """
    Convenience function to generate embedding for text.
    
    Args:
        text: Input text
        
    Returns:
        Embedding vector
    """
    engineer = FeatureEngineer()
    return engineer.generate_embedding(text)
