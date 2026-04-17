# ==============================================================
# MODULE 4 – PURE ML PREDICTION LAYER (For Flask Backend)
# ==============================================================

import os
import numpy as np
import pandas as pd
import joblib
import xml.etree.ElementTree as ET
from tensorflow.keras.models import load_model
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XMLParserPredictor:
    def __init__(self, model_dir="."):
        # Use absolute path to avoid confusion
        self.model_dir = os.path.abspath(model_dir)
        print(f"🔧 Initializing XMLParserPredictor with model_dir: {self.model_dir}")
        print(f"📁 Current working directory: {os.getcwd()}")
        
        self.scaler = "feature_scalar.joblib"
        self.ann_feature_extractor = "hybrid_ann_features_extractor.keras"
        self.svm_classifier = "hybrid_svm_classifier.joblib"
        self.algo_names = {0: 'DOM', 1: 'JDOM', 2: 'PXTG', 3: 'SAX', 4: 'StAX'}
        
        # Load models
        self._load_models()

    def _load_models(self):
        """Load trained models - FIXED for comma filenames"""
        try:
            print(f"🔍 Looking for model files in: {os.path.abspath(self.model_dir)}")
            
            # Get current directory
            current_dir = os.path.abspath(self.model_dir)
            
            # List all files to see what we have
            all_files = os.listdir(current_dir)
            print(f"📁 All files in directory: {all_files}")
            
            # Your actual file names (with commas)
            scaler_file = "feature_scaler.joblib"
            ann_file = "hybrid_ann_features_extractor.keras"
            svm_file = "hybrid_svm_classifier.joblib"
            
            # Build full paths
            scaler_path = os.path.join(current_dir, scaler_file)
            ann_path = os.path.join(current_dir, ann_file)
            svm_path = os.path.join(current_dir, svm_file)
            
            print(f"🔍 Checking specific files:")
            print(f"   - Scaler: {scaler_path} → Exists: {os.path.exists(scaler_path)}")
            print(f"   - ANN: {ann_path} → Exists: {os.path.exists(ann_path)}")
            print(f"   - SVM: {svm_path} → Exists: {os.path.exists(svm_path)}")
            
            # Check if all files exist
            if not all([os.path.exists(scaler_path), os.path.exists(ann_path), os.path.exists(svm_path)]):
                missing = []
                if not os.path.exists(scaler_path): missing.append(scaler_file)
                if not os.path.exists(ann_path): missing.append(ann_file)
                if not os.path.exists(svm_path): missing.append(svm_file)
                raise FileNotFoundError(f"Missing files: {missing}")
            
            # Load models directly (no more complex searching)
            print("🔄 Loading models...")
            
            # Load scaler
            print(f"📦 Loading scaler: {scaler_file}")
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully")
            
            # Load ANN
            print(f"📦 Loading ANN: {ann_file}")
            self.ann_feature_extractor = load_model(ann_path, compile=False)
            print("✅ ANN loaded successfully")
            
            # Load SVM
            print(f"📦 Loading SVM: {svm_file}")
            self.svm_classifier = joblib.load(svm_path)
            print("✅ SVM loaded successfully")
            
            print("🎉 ALL MODELS LOADED SUCCESSFULLY!")
            print(f"   - Scaler type: {type(self.scaler)}")
            print(f"   - ANN type: {type(self.ann_feature_extractor)}")
            print(f"   - SVM type: {type(self.svm_classifier)}")
            
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            logger.error(f"Error loading model components: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error loading model components: {e}")

    def extract_xml_features(self, xml_file_path, cpu_cores):
        """Extract features from XML file"""
        try:
            file_size_mb = os.path.getsize(xml_file_path) / (1024.0 * 1024.0)
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            num_tags = 0
            num_elements = 0
            num_attributes = 0
            max_depth = 0

            def traverse(el, depth):
                nonlocal num_tags, num_elements, num_attributes, max_depth
                num_tags += 1
                num_elements += 1
                num_attributes += len(el.attrib)
                max_depth = max(max_depth, depth)
                for child in el:
                    traverse(child, depth + 1)

            traverse(root, 1)

            features = {
                "File Size (MB)": round(file_size_mb, 3),
                "No of Tags": num_tags,
                "XML Depth": max_depth,
                "No of Attributes": num_attributes,
                "No of Elements": num_elements,
                "CPU Cores": cpu_cores,
                "Memory Usage (MB)": 512
            }
            
            # Convert to display format
            display_features = {
                'File Size (MB)': features["File Size (MB)"],
                'CPU Cores': cpu_cores,
                'Total Tags': num_tags,
                'Total Elements': num_elements,
                'Total Attributes': num_attributes,
                'Structure Depth': max_depth,
                'Attributes per Element': round(num_attributes / num_elements if num_elements > 0 else 0, 3),
                'Memory Usage (MB)': 512
            }
            
            logger.info(f"📊 Extracted features: {display_features}")
            return features, display_features
            
        except Exception as e:
            logger.error(f"❌ Error parsing XML: {e}")
            raise Exception(f"Error parsing XML: {e}")

    def engineer_features(self, df):
        """Apply feature engineering (same as training)"""
        df = df.copy()

        # Original feature engineering from Module 3
        df["tags_per_mb"] = df["No of Tags"] / (df["File Size (MB)"] + 0.01)
        df["elements_per_mb"] = df["No of Elements"] / (df["File Size (MB)"] + 0.01)
        df["attrs_per_element"] = df["No of Attributes"] / (df["No of Elements"] + 1)
        df["memory_efficiency"] = df["Memory Usage (MB)"] / (df["File Size (MB)"] + 0.01)
        df["memory_per_core"] = df["Memory Usage (MB)"] / df["CPU Cores"]
        df["structural_complexity"] = (df["XML Depth"] * df["No of Attributes"]) / (df["No of Elements"] + 1)
        df["data_density"] = (df["No of Tags"] + df["No of Elements"]) / (df["File Size (MB)"] + 0.01)
        df["depth_complexity"] = df["XML Depth"] / (df["File Size (MB)"] + 1)
        df["cores_x_filesize"] = df["CPU Cores"] * df["File Size (MB)"]
        df["cores_x_memory"] = df["CPU Cores"] * df["Memory Usage (MB)"]
        df["filesize_squared"] = df["File Size (MB)"] ** 2
        df["cores_squared"] = df["CPU Cores"] ** 2
        df["log_filesize"] = np.log1p(df["File Size (MB)"])
        df["log_tags"] = np.log1p(df["No of Tags"])
        df["log_elements"] = np.log1p(df["No of Elements"])

        def categorize_size(x):
            if x <= 4: return 0
            elif x <= 25: return 1
            elif x <= 60: return 2
            elif x <= 100: return 3
            else: return 4

        def categorize_core(x):
            if x <= 2: return 0
            elif x <= 8: return 1
            else: return 2

        df["size_category"] = df["File Size (MB)"].apply(categorize_size)
        df["core_category"] = df["CPU Cores"].apply(categorize_core)
        df["size_core_combo"] = (df["size_category"] * 10) + df["core_category"]

        logger.info(f"🔧 Engineered {len(df.columns)} total features")
        return df

    def predict(self, xml_file_path, cpu_cores):
        """Make prediction using ML models"""
        try:
            logger.info(f"🔮 Starting prediction for: {xml_file_path} with {cpu_cores} cores")
            
            # Step 1: Extract base features
            base_features, display_features = self.extract_xml_features(xml_file_path, cpu_cores)

            # Step 2: Engineer features
            df = pd.DataFrame([base_features])
            engineered = self.engineer_features(df)

            # Step 3: Scale features
            scaled = self.scaler.transform(engineered)

            # Step 4: Extract ANN latent features
            ann_features = self.ann_feature_extractor.predict(scaled, verbose=0)

            # Step 5: SVM prediction
            probabilities = self.svm_classifier.predict_proba(ann_features)[0]
            predicted_class = np.argmax(probabilities)
            predicted_algo = self.algo_names[predicted_class]
            confidence = probabilities[predicted_class]

            result = {
                "algorithm": predicted_algo,
                "confidence": float(confidence),
                "probabilities": {self.algo_names[i]: float(prob) for i, prob in enumerate(probabilities)},
                "extracted_features": display_features  # Use display format
            }
            
            logger.info(f"🎯 Prediction result: {predicted_algo} (confidence: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def get_parser_code(self, algorithm):
        """Get parser implementation code for the recommended algorithm"""
        parser_codes = {
            'DOM': '''# DOM Parser - Best for small files
import xml.dom.minidom

def parse_xml_dom(file_path):
    """Parse XML using DOM parser"""
    try:
        # Parse the XML file
        dom_tree = xml.dom.minidom.parse(file_path)
        root = dom_tree.documentElement
        
        # Process the XML data
        print(f"Root element: {root.tagName}")
        
        # Get all elements
        elements = root.getElementsByTagName('*')
        print(f"Total elements: {len(elements)}")
        
        # Example: Process specific elements
        for element in elements:
            if element.nodeType == element.ELEMENT_NODE:
                print(f"Element: {element.tagName}")
                if element.hasChildNodes():
                    for child in element.childNodes:
                        if child.nodeType == child.TEXT_NODE and child.data.strip():
                            print(f"  Content: {child.data.strip()}")
        
        return dom_tree
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

# Usage
if __name__ == "__main__":
    result = parse_xml_dom("your_file.xml")
    if result:
        print("✅ XML parsed successfully using DOM parser")''',

            'JDOM': '''# JDOM-style Parser (Python equivalent)
import xml.etree.ElementTree as ET

def parse_xml_jdom_style(file_path):
    """Parse XML using JDOM-style approach (ElementTree)"""
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")
        
        # Recursive function to process all elements
        def process_element(element, depth=0):
            indent = "  " * depth
            print(f"{indent}Element: {element.tag}")
            
            # Print attributes
            if element.attrib:
                print(f"{indent}  Attributes: {element.attrib}")
            
            # Print text content
            if element.text and element.text.strip():
                print(f"{indent}  Content: {element.text.strip()}")
            
            # Process children
            for child in element:
                process_element(child, depth + 1)
        
        # Start processing from root
        process_element(root)
        return tree
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

# Usage
if __name__ == "__main__":
    result = parse_xml_jdom_style("your_file.xml")
    if result:
        print("✅ XML parsed successfully using JDOM-style parser")''',

            'SAX': '''# SAX Parser - Best for large files (memory efficient)
import xml.sax

class XMLHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.current_element = ""
        self.content = ""
        self.depth = 0
        
    def startElement(self, name, attrs):
        self.current_element = name
        self.content = ""
        print("  " * self.depth + f"Start: {name}")
        
        # Print attributes
        if attrs:
            for attr_name, attr_value in attrs.items():
                print("  " * (self.depth + 1) + f"Attr: {attr_name} = {attr_value}")
        
        self.depth += 1
        
    def endElement(self, name):
        self.depth -= 1
        if self.content.strip():
            print("  " * self.depth + f"Content: {self.content.strip()}")
        print("  " * self.depth + f"End: {name}")
        self.content = ""
        
    def characters(self, content):
        self.content += content

def parse_xml_sax(file_path):
    """Parse XML using SAX parser (event-based)"""
    try:
        # Create a parser
        parser = xml.sax.make_parser()
        
        # Turn off namespaces
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        
        # Override the default ContextHandler
        handler = XMLHandler()
        parser.setContentHandler(handler)
        
        print("Starting SAX parsing...")
        parser.parse(file_path)
        print("SAX parsing completed!")
        
        return True
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return False

# Usage
if __name__ == "__main__":
    success = parse_xml_sax("your_file.xml")
    if success:
        print("✅ XML parsed successfully using SAX parser")''',

            'StAX': '''# StAX-style Parser (Python equivalent using iterative parsing)
import xml.etree.ElementTree as ET

def parse_xml_stax_style(file_path):
    """Parse XML using StAX-style iterative parsing"""
    try:
        print("Starting StAX-style iterative parsing...")
        
        # Use iterparse for memory-efficient iterative parsing
        context = ET.iterparse(file_path, events=("start", "end"))
        
        # Turn it into an iterator
        context = iter(context)
        
        # Get the root element
        event, root = next(context)
        
        element_stack = []
        
        for event, elem in context:
            if event == "start":
                element_stack.append(elem.tag)
                print("  " * (len(element_stack) - 1) + f"Start: {elem.tag}")
                
                # Print attributes
                if elem.attrib:
                    for attr, value in elem.attrib.items():
                        print("  " * len(element_stack) + f"Attr: {attr} = {value}")
                        
            elif event == "end":
                # Print text content
                if elem.text and elem.text.strip():
                    print("  " * len(element_stack) + f"Content: {elem.text.strip()}")
                
                print("  " * (len(element_stack) - 1) + f"End: {elem.tag}")
                element_stack.pop()
                
                # Clear processed elements to save memory
                elem.clear()
        
        print("StAX-style parsing completed!")
        return True
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return False

# Usage
if __name__ == "__main__":
    success = parse_xml_stax_style("your_file.xml")
    if success:
        print("✅ XML parsed successfully using StAX-style parser")''',

            'PXTG': '''# PXTG Parser (Custom Python XML to Graph Transformer)
import xml.etree.ElementTree as ET
import networkx as nx

def parse_xml_pxtg(file_path):
    """Parse XML and transform to graph representation"""
    try:
        # Parse XML
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Create a graph
        graph = nx.Graph()
        
        def build_graph(element, parent=None, depth=0):
            """Recursively build graph from XML structure"""
            node_id = f"{element.tag}_{id(element)}"
            
            # Add current node
            graph.add_node(node_id, tag=element.tag, attributes=element.attrib)
            
            # Connect to parent
            if parent:
                graph.add_edge(parent, node_id)
            
            # Process children
            for child in element:
                build_graph(child, node_id, depth + 1)
            
            return graph
        
        # Build the graph
        xml_graph = build_graph(root)
        
        print(f"Graph nodes: {xml_graph.number_of_nodes()}")
        print(f"Graph edges: {xml_graph.number_of_edges()}")
        
        # Print graph structure
        print("\\nGraph structure:")
        for node in xml_graph.nodes(data=True):
            print(f"Node: {node[0]}, Data: {node[1]}")
        
        return xml_graph
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

# Usage
if __name__ == "__main__":
    graph = parse_xml_pxtg("your_file.xml")
    if graph:
        print("✅ XML parsed successfully using PXTG parser")
        print("💡 This creates a graph representation of your XML structure")'''
        }
        
        return parser_codes.get(algorithm, "# Parser code not available for this algorithm")