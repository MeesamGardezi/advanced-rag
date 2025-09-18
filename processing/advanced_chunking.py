import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Multi-version LlamaIndex imports with fallback
try:
    # Try newer version imports (0.10.x+)
    from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
    from llama_index.core import Document, TextNode
    from llama_index.core.schema import MetadataMode, NodeRelationship
    LLAMA_INDEX_AVAILABLE = True
    LLAMA_INDEX_VERSION = "new"
except ImportError:
    try:
        # Try older version imports (0.8.x - 0.9.x)
        from llama_index.node_parser import HierarchicalNodeParser, SentenceSplitter
        from llama_index import Document, TextNode
        from llama_index.schema import MetadataMode, NodeRelationship
        LLAMA_INDEX_AVAILABLE = True
        LLAMA_INDEX_VERSION = "old"
    except ImportError:
        try:
            # Try even older version patterns
            from llama_index.text_splitter import SentenceSplitter
            from llama_index.node_parser import SimpleNodeParser as HierarchicalNodeParser
            from llama_index import Document, TextNode
            from llama_index.schema import MetadataMode
            
            # Create dummy NodeRelationship for older versions
            class NodeRelationship:
                PARENT = "parent"
                CHILD = "child"
                NEXT = "next"
                PREVIOUS = "previous"
            
            LLAMA_INDEX_AVAILABLE = True
            LLAMA_INDEX_VERSION = "legacy"
        except ImportError:
            # Create fallback classes when LlamaIndex is not available
            LLAMA_INDEX_AVAILABLE = False
            LLAMA_INDEX_VERSION = "none"
            
            class Document:
                def __init__(self, text, metadata=None):
                    self.text = text
                    self.metadata = metadata or {}
                    self.id = f"doc_{hash(text[:100])}"
            
            class TextNode:
                def __init__(self, text="", metadata=None):
                    self.text = text
                    self.metadata = metadata or {}
                    self.id = f"node_{hash(text[:100])}"
                    self.node_id = self.id
                    self.relationships = {}
                
                def get_content(self, metadata_mode=None):
                    return self.text
            
            class HierarchicalNodeParser:
                @classmethod
                def from_defaults(cls, chunk_sizes=None, chunk_overlap=200):
                    return cls()
                
                def get_nodes_from_documents(self, documents):
                    return []
            
            class SentenceSplitter:
                def __init__(self, chunk_size=1000, chunk_overlap=200):
                    self.chunk_size = chunk_size
                    self.chunk_overlap = chunk_overlap
                
                def get_nodes_from_documents(self, documents):
                    return []
            
            class MetadataMode:
                NONE = "none"
                LLM = "llm"
                EMBED = "embed"
                ALL = "all"
            
            class NodeRelationship:
                PARENT = "parent"
                CHILD = "child"
                NEXT = "next"
                PREVIOUS = "previous"
            
            print("⚠️  LlamaIndex not available. Advanced chunking will use fallback methods.")

from core.config import config

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Enhanced metadata for document chunks"""
    chunk_id: str
    parent_id: Optional[str]
    chunk_type: str  # 'parent', 'child', 'leaf', 'section', 'fallback'
    chunk_level: int  # 0 = top level, 1 = child, 2 = grandchild
    chunk_index: int  # Position within parent
    original_document_id: str
    construction_category: str
    section_title: Optional[str]
    data_type: str
    job_context: Dict[str, Any]
    chunk_size: int
    relationships: Dict[str, List[str]]  # parent, children, siblings
    processing_version: str = "2.0"
    llama_index_version: str = LLAMA_INDEX_VERSION

@dataclass
class ProcessedChunk:
    """A processed document chunk with enhanced metadata"""
    id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

class ConstructionDocumentAnalyzer:
    """Analyze construction documents for better chunking"""
    
    def __init__(self):
        # Construction document section patterns
        self.section_patterns = {
            'project_info': [
                r'project\s+information', r'project\s+details', r'job\s+summary',
                r'project\s+title', r'client\s+information', r'job\s+description'
            ],
            'cost_breakdown': [
                r'cost\s+breakdown', r'budget\s+summary', r'financial\s+summary',
                r'cost\s+analysis', r'expense\s+report', r'cost\s+summary'
            ],
            'materials': [
                r'materials?\s+list', r'material\s+costs?', r'supplies',
                r'material\s+breakdown', r'equipment', r'tools'
            ],
            'labor': [
                r'labor\s+costs?', r'workforce', r'manpower',
                r'labor\s+breakdown', r'crew\s+costs?', r'hours'
            ],
            'schedule': [
                r'schedule', r'timeline', r'project\s+plan',
                r'milestones?', r'deadlines?', r'duration', r'phases?'
            ],
            'specifications': [
                r'specifications?', r'technical\s+requirements',
                r'specs', r'requirements', r'standards?'
            ],
            'compliance': [
                r'code\s+compliance', r'regulations?', r'permits?',
                r'inspections?', r'approvals?', r'codes?'
            ]
        }
        
        # Cost code patterns for better categorization
        self.cost_code_pattern = re.compile(r'\b\d{3}[SMLO]?\b')
        
        # Construction-specific entity patterns
        self.entity_patterns = {
            'monetary': re.compile(r'\$[\d,]+(?:\.\d{2})?'),
            'measurements': re.compile(r'\b\d+(?:\.\d+)?\s*(?:sq\s*ft|linear\s*ft|cubic\s*yard|ton|gallon|hour|lf|sf|cy|ft|yards?|inches?)\b', re.IGNORECASE),
            'dates': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'),
            'percentages': re.compile(r'\b\d+(?:\.\d+)?%\b'),
            'areas': re.compile(r'\b(?:kitchen|bathroom|bedroom|living\s+room|garage|basement|attic|office|lobby|hallway)\b', re.IGNORECASE),
            'trades': re.compile(r'\b(?:electrical|plumbing|hvac|mechanical|structural|concrete|framing|roofing|flooring|painting|drywall)\b', re.IGNORECASE)
        }
    
    def analyze_document(self, content: str, job_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document structure and content for optimal chunking"""
        analysis = {
            'document_type': self._classify_document_type(content, job_metadata),
            'sections': self._identify_sections(content),
            'entities': self._extract_entities(content),
            'complexity_score': self._calculate_complexity(content),
            'construction_category': self._categorize_construction_content(content, job_metadata),
            'chunking_strategy': 'hierarchical',
            'word_count': len(content.split()),
            'line_count': len(content.split('\n')),
            'has_structured_data': self._has_structured_data(content)
        }
        
        # Determine optimal chunking strategy based on analysis
        if analysis['complexity_score'] > 0.8:
            analysis['chunking_strategy'] = 'fine_grained'
        elif analysis['document_type'] in ['schedule', 'estimate'] or analysis['has_structured_data']:
            analysis['chunking_strategy'] = 'structured'
        elif analysis['word_count'] > 5000:
            analysis['chunking_strategy'] = 'hierarchical'
        else:
            analysis['chunking_strategy'] = 'simple'
        
        return analysis
    
    def _classify_document_type(self, content: str, job_metadata: Dict[str, Any]) -> str:
        """Classify the type of construction document"""
        content_lower = content.lower()
        data_type = job_metadata.get('data_type', '')
        
        # Direct from metadata
        if data_type in ['consumed', 'estimate', 'schedule']:
            return data_type
        
        # Pattern-based classification with scoring
        type_scores = {}
        for section_type, patterns in self.section_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            if score > 0:
                type_scores[section_type] = score
        
        if not type_scores:
            return 'general'
        
        # Return the type with the highest score
        best_type = max(type_scores, key=type_scores.get)
        return best_type
    
    def _identify_sections(self, content: str) -> List[Dict[str, Any]]:
        """Identify document sections for structured chunking"""
        sections = []
        content_lower = content.lower()
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, content_lower))
                for match in matches:
                    # Look for the full line containing the match
                    start_line = content.rfind('\n', 0, match.start()) + 1
                    end_line = content.find('\n', match.end())
                    if end_line == -1:
                        end_line = len(content)
                    
                    sections.append({
                        'type': section_type,
                        'start': start_line,
                        'end': end_line,
                        'title': content[start_line:end_line].strip(),
                        'pattern': pattern,
                        'confidence': 1.0  # Could be enhanced with ML scoring
                    })
        
        # Remove overlapping sections and sort by position
        sections = self._remove_overlapping_sections(sections)
        sections.sort(key=lambda s: s['start'])
        
        return sections
    
    def _remove_overlapping_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping sections, keeping the one with higher confidence"""
        if not sections:
            return sections
        
        # Sort by start position
        sections.sort(key=lambda s: s['start'])
        
        filtered_sections = [sections[0]]
        
        for section in sections[1:]:
            last_section = filtered_sections[-1]
            
            # Check for overlap
            if section['start'] < last_section['end']:
                # Overlap detected - keep the one with higher confidence
                if section['confidence'] > last_section['confidence']:
                    filtered_sections[-1] = section
                # else keep the existing one
            else:
                filtered_sections.append(section)
        
        return filtered_sections
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract construction-specific entities"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(content)
            entities[entity_type] = list(set(matches))  # Remove duplicates
        
        # Extract cost codes
        cost_codes = self.cost_code_pattern.findall(content)
        entities['cost_codes'] = list(set(cost_codes))
        
        return entities
    
    def _has_structured_data(self, content: str) -> bool:
        """Check if content has structured data (tables, lists, etc.)"""
        # Look for table-like structures
        lines = content.split('\n')
        
        # Check for tabular data
        tab_separated_lines = sum(1 for line in lines if '\t' in line or '|' in line)
        if tab_separated_lines > 3:
            return True
        
        # Check for numbered lists
        numbered_lists = sum(1 for line in lines if re.match(r'^\s*\d+\.', line.strip()))
        if numbered_lists > 5:
            return True
        
        # Check for bullet points
        bullet_points = sum(1 for line in lines if re.match(r'^\s*[-*•]', line.strip()))
        if bullet_points > 5:
            return True
        
        return False
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate document complexity score for chunking decisions"""
        factors = {
            'length': min(len(content) / 10000, 1.0),  # Normalized by 10k characters
            'technical_terms': len(re.findall(r'\b(?:specification|requirement|compliance|regulation|code|standard)\b', content, re.IGNORECASE)) / 100,
            'numerical_data': len(re.findall(r'\$[\d,]+|\b\d+(?:\.\d+)?%|\b\d+\s*(?:sq\s*ft|hour)', content, re.IGNORECASE)) / 50,
            'structure_indicators': len(re.findall(r'^\s*(?:\d+\.|\*|\-)', content, re.MULTILINE)) / 20,
            'entities': sum(len(entities) for entities in self._extract_entities(content).values()) / 100
        }
        
        # Weighted complexity score
        complexity = (
            factors['length'] * 0.15 +
            factors['technical_terms'] * 0.25 +
            factors['numerical_data'] * 0.25 +
            factors['structure_indicators'] * 0.15 +
            factors['entities'] * 0.20
        )
        
        return min(complexity, 1.0)
    
    def _categorize_construction_content(self, content: str, job_metadata: Dict[str, Any]) -> str:
        """Categorize content for construction domain"""
        content_lower = content.lower()
        
        # Score-based categorization
        category_scores = {
            'materials': 0,
            'labor': 0,
            'subcontractors': 0,
            'compliance': 0,
            'scheduling': 0,
            'costs': 0
        }
        
        # Material indicators
        material_terms = ['material', 'lumber', 'concrete', 'steel', 'electrical', 'supplies', 'equipment']
        category_scores['materials'] = sum(content_lower.count(term) for term in material_terms)
        
        # Labor indicators
        labor_terms = ['labor', 'crew', 'worker', 'manpower', 'hours', 'wages']
        category_scores['labor'] = sum(content_lower.count(term) for term in labor_terms)
        
        # Subcontractor indicators
        sub_terms = ['subcontractor', 'contractor', 'vendor', 'supplier']
        category_scores['subcontractors'] = sum(content_lower.count(term) for term in sub_terms)
        
        # Compliance indicators
        compliance_terms = ['permit', 'code', 'regulation', 'compliance', 'inspection', 'approval']
        category_scores['compliance'] = sum(content_lower.count(term) for term in compliance_terms)
        
        # Scheduling indicators
        schedule_terms = ['schedule', 'timeline', 'deadline', 'duration', 'phase', 'milestone']
        category_scores['scheduling'] = sum(content_lower.count(term) for term in schedule_terms)
        
        # Cost indicators
        cost_terms = ['cost', 'budget', 'expense', 'price', 'total', 'amount']
        category_scores['costs'] = sum(content_lower.count(term) for term in cost_terms)
        
        # Return category with highest score, or 'general' if all scores are low
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'

class AdvancedChunking:
    """Advanced semantic chunking for construction documents"""
    
    def __init__(self):
        self.use_llamaindex = LLAMA_INDEX_AVAILABLE and config.processing.enable_semantic_chunking
        self.document_analyzer = ConstructionDocumentAnalyzer()
        
        # Initialize parsers based on available version
        if self.use_llamaindex:
            try:
                if LLAMA_INDEX_VERSION in ["new", "old"]:
                    self.hierarchical_parser = HierarchicalNodeParser.from_defaults(
                        chunk_sizes=config.processing.chunk_sizes,
                        chunk_overlap=config.processing.chunk_overlap
                    )
                    
                    self.sentence_splitter = SentenceSplitter(
                        chunk_size=config.processing.chunk_sizes[0],
                        chunk_overlap=config.processing.chunk_overlap
                    )
                else:
                    # Legacy version
                    self.hierarchical_parser = HierarchicalNodeParser()
                    self.sentence_splitter = SentenceSplitter(
                        chunk_size=config.processing.chunk_sizes[0],
                        chunk_overlap=config.processing.chunk_overlap
                    )
                
                logger.info(f"✅ LlamaIndex {LLAMA_INDEX_VERSION} version initialized for advanced chunking")
            except Exception as e:
                logger.warning(f"⚠️  LlamaIndex initialization failed: {e}. Using fallback chunking.")
                self.use_llamaindex = False
        
        logger.info(f"🔧 Advanced chunking initialized (LlamaIndex: {self.use_llamaindex})")
    
    def chunk_construction_document(self, 
                                  content: str,
                                  job_metadata: Dict[str, Any],
                                  document_id: Optional[str] = None) -> List[ProcessedChunk]:
        """Chunk construction document with domain-specific optimization"""
        
        if not content.strip():
            logger.warning("Empty content provided for chunking")
            return []
        
        # Analyze document first
        analysis = self.document_analyzer.analyze_document(content, job_metadata)
        
        document_id = document_id or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.debug(f"📋 Document analysis: {analysis['document_type']}, strategy: {analysis['chunking_strategy']}, complexity: {analysis['complexity_score']:.2f}")
        
        if self.use_llamaindex:
            return self._semantic_chunking(content, job_metadata, analysis, document_id)
        else:
            return self._fallback_chunking(content, job_metadata, analysis, document_id)
    
    def _semantic_chunking(self, 
                          content: str,
                          job_metadata: Dict[str, Any],
                          analysis: Dict[str, Any],
                          document_id: str) -> List[ProcessedChunk]:
        """Perform semantic chunking using LlamaIndex"""
        try:
            # Create LlamaIndex document
            doc = Document(
                text=content,
                metadata={
                    **job_metadata,
                    'document_id': document_id,
                    'analysis': analysis
                }
            )
            
            # Choose chunking strategy based on analysis
            if analysis['chunking_strategy'] == 'structured':
                chunks = self._structured_chunking(doc, analysis, document_id)
            elif analysis['chunking_strategy'] == 'fine_grained':
                chunks = self._fine_grained_chunking(doc, analysis, document_id)
            elif analysis['chunking_strategy'] == 'hierarchical':
                chunks = self._hierarchical_chunking(doc, analysis, document_id)
            else:
                chunks = self._simple_chunking(doc, analysis, document_id)
            
            logger.info(f"✅ Created {len(chunks)} semantic chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Semantic chunking failed: {e}. Using fallback chunking.")
            return self._fallback_chunking(content, job_metadata, analysis, document_id)
    
    def _hierarchical_chunking(self, 
                              doc: Document,
                              analysis: Dict[str, Any],
                              document_id: str) -> List[ProcessedChunk]:
        """Hierarchical chunking using LlamaIndex"""
        try:
            # Parse document into hierarchical nodes
            nodes = self.hierarchical_parser.get_nodes_from_documents([doc])
            
            processed_chunks = []
            
            for i, node in enumerate(nodes):
                # Determine chunk level and relationships
                chunk_level = 0  # Default to top level
                parent_id = None
                
                # Handle relationships based on LlamaIndex version
                relationships = {}
                try:
                    if hasattr(node, 'relationships') and node.relationships:
                        relationships = self._extract_relationships(node)
                        # Check for parent relationship
                        if 'parent' in relationships:
                            parent_id = relationships['parent'][0] if relationships['parent'] else None
                            chunk_level = 1
                except Exception as e:
                    logger.debug(f"Could not extract relationships: {e}")
                
                # Create enhanced metadata
                chunk_metadata = ChunkMetadata(
                    chunk_id=str(getattr(node, 'node_id', f"{document_id}_chunk_{i}")),
                    parent_id=parent_id,
                    chunk_type='parent' if parent_id is None else 'child',
                    chunk_level=chunk_level,
                    chunk_index=i,
                    original_document_id=document_id,
                    construction_category=analysis['construction_category'],
                    section_title=self._extract_section_title(node.text),
                    data_type=doc.metadata.get('data_type', 'unknown'),
                    job_context={
                        'job_name': doc.metadata.get('job_name', ''),
                        'company_id': doc.metadata.get('company_id', ''),
                        'job_id': doc.metadata.get('job_id', '')
                    },
                    chunk_size=len(node.text),
                    relationships=relationships,
                    llama_index_version=LLAMA_INDEX_VERSION
                )
                
                processed_chunk = ProcessedChunk(
                    id=str(getattr(node, 'node_id', f"{document_id}_chunk_{i}")),
                    content=node.get_content(metadata_mode=MetadataMode.NONE) if hasattr(node, 'get_content') else node.text,
                    metadata=chunk_metadata
                )
                
                processed_chunks.append(processed_chunk)
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Hierarchical chunking failed: {e}")
            return []
    
    def _simple_chunking(self, doc: Document, analysis: Dict[str, Any], document_id: str) -> List[ProcessedChunk]:
        """Simple chunking for smaller documents"""
        try:
            nodes = self.sentence_splitter.get_nodes_from_documents([doc])
            
            processed_chunks = []
            for i, node in enumerate(nodes):
                chunk_id = f"{document_id}_simple_{i}"
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    parent_id=document_id,
                    chunk_type='simple',
                    chunk_level=1,
                    chunk_index=i,
                    original_document_id=document_id,
                    construction_category=analysis['construction_category'],
                    section_title=None,
                    data_type=doc.metadata.get('data_type', 'unknown'),
                    job_context={
                        'job_name': doc.metadata.get('job_name', ''),
                        'company_id': doc.metadata.get('company_id', ''),
                        'job_id': doc.metadata.get('job_id', '')
                    },
                    chunk_size=len(node.text),
                    relationships={},
                    llama_index_version=LLAMA_INDEX_VERSION
                )
                
                chunk = ProcessedChunk(
                    id=chunk_id,
                    content=node.get_content(metadata_mode=MetadataMode.NONE) if hasattr(node, 'get_content') else node.text,
                    metadata=chunk_metadata
                )
                
                processed_chunks.append(chunk)
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Simple chunking failed: {e}")
            return []
    
    def _structured_chunking(self,
                           doc: Document,
                           analysis: Dict[str, Any],
                           document_id: str) -> List[ProcessedChunk]:
        """Structured chunking based on document sections"""
        sections = analysis.get('sections', [])
        
        if not sections:
            # Fall back to hierarchical if no sections found
            return self._hierarchical_chunking(doc, analysis, document_id)
        
        processed_chunks = []
        content = doc.text
        
        # Process each identified section
        for i, section in enumerate(sections):
            # Determine section boundaries
            start = section['start']
            end = sections[i + 1]['start'] if i + 1 < len(sections) else len(content)
            
            section_content = content[start:end].strip()
            
            if len(section_content) < 50:  # Skip very short sections
                continue
            
            # If section is too long, split it further
            if len(section_content) > config.processing.chunk_sizes[0]:
                sub_chunks = self._split_large_section(section_content, section, document_id, i, doc)
                processed_chunks.extend(sub_chunks)
            else:
                # Create single chunk for section
                chunk_id = f"{document_id}_section_{i}"
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    parent_id=document_id,
                    chunk_type='section',
                    chunk_level=1,
                    chunk_index=i,
                    original_document_id=document_id,
                    construction_category=section['type'],
                    section_title=section['title'],
                    data_type=doc.metadata.get('data_type', 'unknown'),
                    job_context={
                        'job_name': doc.metadata.get('job_name', ''),
                        'company_id': doc.metadata.get('company_id', ''),
                        'job_id': doc.metadata.get('job_id', '')
                    },
                    chunk_size=len(section_content),
                    relationships={'siblings': [f"{document_id}_section_{j}" for j in range(len(sections)) if j != i]},
                    llama_index_version=LLAMA_INDEX_VERSION
                )
                
                chunk = ProcessedChunk(
                    id=chunk_id,
                    content=section_content,
                    metadata=chunk_metadata
                )
                
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _fine_grained_chunking(self,
                             doc: Document,
                             analysis: Dict[str, Any],
                             document_id: str) -> List[ProcessedChunk]:
        """Fine-grained chunking for complex documents"""
        try:
            # Use smaller chunk sizes for complex documents
            fine_grained_splitter = SentenceSplitter(
                chunk_size=config.processing.chunk_sizes[-1],  # Use smallest chunk size
                chunk_overlap=config.processing.chunk_overlap
            )
            
            nodes = fine_grained_splitter.get_nodes_from_documents([doc])
            
            processed_chunks = []
            
            for i, node in enumerate(nodes):
                chunk_id = f"{document_id}_fine_{i}"
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    parent_id=document_id,
                    chunk_type='fine_grained',
                    chunk_level=2,
                    chunk_index=i,
                    original_document_id=document_id,
                    construction_category=analysis['construction_category'],
                    section_title=None,
                    data_type=doc.metadata.get('data_type', 'unknown'),
                    job_context={
                        'job_name': doc.metadata.get('job_name', ''),
                        'company_id': doc.metadata.get('company_id', ''),
                        'job_id': doc.metadata.get('job_id', '')
                    },
                    chunk_size=len(node.text),
                    relationships={'siblings': [f"{document_id}_fine_{j}" for j in range(max(0, i-2), min(len(nodes), i+3)) if j != i]},
                    llama_index_version=LLAMA_INDEX_VERSION
                )
                
                chunk = ProcessedChunk(
                    id=chunk_id,
                    content=node.get_content(metadata_mode=MetadataMode.NONE) if hasattr(node, 'get_content') else node.text,
                    metadata=chunk_metadata
                )
                
                processed_chunks.append(chunk)
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Fine-grained chunking failed: {e}")
            return []
    
    def _split_large_section(self,
                           content: str,
                           section: Dict[str, Any],
                           document_id: str,
                           section_index: int,
                           doc: Document) -> List[ProcessedChunk]:
        """Split large sections into manageable chunks"""
        try:
            splitter = SentenceSplitter(
                chunk_size=config.processing.chunk_sizes[1],  # Use medium chunk size
                chunk_overlap=config.processing.chunk_overlap
            )
            
            # Create temporary document for splitting
            temp_doc = Document(text=content)
            nodes = splitter.get_nodes_from_documents([temp_doc])
            
            sub_chunks = []
            
            for i, node in enumerate(nodes):
                chunk_id = f"{document_id}_section_{section_index}_sub_{i}"
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    parent_id=f"{document_id}_section_{section_index}",
                    chunk_type='sub_section',
                    chunk_level=2,
                    chunk_index=i,
                    original_document_id=document_id,
                    construction_category=section['type'],
                    section_title=section['title'],
                    data_type=doc.metadata.get('data_type', 'unknown'),
                    job_context={
                        'job_name': doc.metadata.get('job_name', ''),
                        'company_id': doc.metadata.get('company_id', ''),
                        'job_id': doc.metadata.get('job_id', '')
                    },
                    chunk_size=len(node.text),
                    relationships={'parent': f"{document_id}_section_{section_index}"},
                    llama_index_version=LLAMA_INDEX_VERSION
                )
                
                chunk = ProcessedChunk(
                    id=chunk_id,
                    content=node.get_content(metadata_mode=MetadataMode.NONE) if hasattr(node, 'get_content') else node.text,
                    metadata=chunk_metadata
                )
                
                sub_chunks.append(chunk)
            
            return sub_chunks
            
        except Exception as e:
            logger.error(f"❌ Large section splitting failed: {e}")
            return []
    
    def _fallback_chunking(self,
                         content: str,
                         job_metadata: Dict[str, Any],
                         analysis: Dict[str, Any],
                         document_id: str) -> List[ProcessedChunk]:
        """Fallback chunking when LlamaIndex is not available"""
        logger.info("🔧 Using enhanced fallback chunking method")
        
        # Enhanced sentence-based chunking with construction awareness
        sentences = re.split(r'[.!?]+(?:\s|$)', content)
        chunk_size = config.processing.chunk_sizes[0]
        overlap = config.processing.chunk_overlap
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Create chunk
                chunk_id = f"{document_id}_fallback_{chunk_index}"
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    parent_id=document_id,
                    chunk_type='fallback',
                    chunk_level=1,
                    chunk_index=chunk_index,
                    original_document_id=document_id,
                    construction_category=analysis['construction_category'],
                    section_title=self._extract_section_title(current_chunk),
                    data_type=job_metadata.get('data_type', 'unknown'),
                    job_context={
                        'job_name': job_metadata.get('job_name', ''),
                        'company_id': job_metadata.get('company_id', ''),
                        'job_id': job_metadata.get('job_id', '')
                    },
                    chunk_size=len(current_chunk),
                    relationships={},
                    llama_index_version="fallback"
                )
                
                chunk = ProcessedChunk(
                    id=chunk_id,
                    content=current_chunk.strip(),
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                
                chunk_index += 1
            else:
                current_chunk += " " + sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = f"{document_id}_fallback_{chunk_index}"
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                parent_id=document_id,
                chunk_type='fallback',
                chunk_level=1,
                chunk_index=chunk_index,
                original_document_id=document_id,
                construction_category=analysis['construction_category'],
                section_title=self._extract_section_title(current_chunk),
                data_type=job_metadata.get('data_type', 'unknown'),
                job_context={
                    'job_name': job_metadata.get('job_name', ''),
                    'company_id': job_metadata.get('company_id', ''),
                    'job_id': job_metadata.get('job_id', '')
                },
                chunk_size=len(current_chunk),
                relationships={},
                llama_index_version="fallback"
            )
            
            chunk = ProcessedChunk(
                id=chunk_id,
                content=current_chunk.strip(),
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
        
        logger.info(f"✅ Created {len(chunks)} fallback chunks")
        return chunks
    
    def _extract_section_title(self, content: str) -> Optional[str]:
        """Extract section title from content"""
        lines = content.strip().split('\n')
        if not lines:
            return None
        
        first_line = lines[0].strip()
        
        # Check if first line looks like a title
        if len(first_line) < 100 and (
            first_line.isupper() or
            re.match(r'^[A-Z][^.!?]*$', first_line) or
            re.match(r'^\d+\.?\s+[A-Z]', first_line) or
            ':' in first_line
        ):
            return first_line
        
        return None
    
    def _extract_relationships(self, node) -> Dict[str, List[str]]:
        """Extract node relationships from LlamaIndex node"""
        relationships = {}
        
        if hasattr(node, 'relationships') and node.relationships:
            try:
                for rel_type, rel_info in node.relationships.items():
                    if hasattr(NodeRelationship, 'PARENT') and rel_type == NodeRelationship.PARENT:
                        relationships['parent'] = [str(rel_info.node_id)]
                    elif hasattr(NodeRelationship, 'CHILD') and rel_type == NodeRelationship.CHILD:
                        relationships.setdefault('children', []).append(str(rel_info.node_id))
                    elif hasattr(NodeRelationship, 'NEXT') and rel_type == NodeRelationship.NEXT:
                        relationships.setdefault('next', []).append(str(rel_info.node_id))
                    elif hasattr(NodeRelationship, 'PREVIOUS') and rel_type == NodeRelationship.PREVIOUS:
                        relationships.setdefault('previous', []).append(str(rel_info.node_id))
            except Exception as e:
                logger.debug(f"Error extracting relationships: {e}")
        
        return relationships

# Global advanced chunking instance
_advanced_chunking = None

def get_advanced_chunking() -> AdvancedChunking:
    """Get or create global advanced chunking instance"""
    global _advanced_chunking
    if _advanced_chunking is None:
        _advanced_chunking = AdvancedChunking()
    return _advanced_chunking