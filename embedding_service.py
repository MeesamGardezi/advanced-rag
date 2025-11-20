import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import time
from datetime import datetime

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
    def create_job_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Convert job data into text suitable for embedding"""
        data_type = job_data.get('data_type', 'consumed')
        
        if data_type == 'consumed':
            return self._create_consumed_text_representation(job_data)
        elif data_type == 'estimate':
            return self._create_estimate_text_representation(job_data)
        elif data_type == 'flooring_estimate':
            return self._create_flooring_estimate_text_representation(job_data)
        elif data_type == 'schedule':
            return self._create_schedule_text_representation(job_data)
        else:
            return f"Unknown data type: {data_type}"
    
    def _create_consumed_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create text representation for consumed cost data"""
        try:
            entries = job_data.get('entries', [])
            job_name = entries[0].get('job', 'Unknown Job') if entries else 'Unknown Job'
            
            text_parts = [
                f"CONSUMED COST DATA",
                f"Job: {job_name}",
                f"Last Updated: {job_data.get('lastUpdated', 'Unknown')}",
                f"Total Entries: {len(entries)}"
            ]
            
            # Group by category
            categories = {}
            total_cost = 0.0
            
            for entry in entries:
                cost_code = entry.get('costCode', 'Unknown')
                amount_str = entry.get('amount', '0')
                
                try:
                    amount = float(amount_str) if amount_str else 0.0
                    total_cost += amount
                except (ValueError, TypeError):
                    amount = 0.0
                
                category = self.categorize_cost_code(cost_code)
                
                if category not in categories:
                    categories[category] = []
                
                categories[category].append({
                    'cost_code': cost_code,
                    'amount': amount
                })
            
            text_parts.append(f"Total Consumed Cost: ${total_cost:,.2f}")
            
            for category, items in categories.items():
                category_total = sum(item['amount'] for item in items)
                text_parts.append(f"\n{category} (${category_total:,.2f}):")
                
                for item in items:
                    if item['amount'] > 0:
                        text_parts.append(f"  - {item['cost_code']}: ${item['amount']:,.2f}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating consumed text: {e}")
            return f"Consumed data processing error: {str(e)}"
    
    def _create_estimate_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create text representation for estimate data"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            
            text_parts = [
                f"ESTIMATE DATA SUMMARY",
                f"Job: {job_name}",
                f"Client: {job_data.get('client_name', 'Unknown')}",
                f"Location: {job_data.get('site_location', 'Unknown')}",
                f"Total Rows: {len(entries)}",
                f""
            ]
            
            # Calculate totals
            total_estimated = sum(float(e.get('total', 0)) for e in entries)
            total_budgeted = sum(float(e.get('budgetedTotal', 0)) for e in entries)
            
            text_parts.extend([
                f"Total Estimated: ${total_estimated:,.2f}",
                f"Total Budgeted: ${total_budgeted:,.2f}",
                f"Variance: ${total_budgeted - total_estimated:,.2f}",
                f""
            ])
            
            # Group by area
            areas = {}
            for entry in entries:
                area = entry.get('area', 'General')
                if area not in areas:
                    areas[area] = []
                areas[area].append(entry)
            
            text_parts.append(f"BREAKDOWN BY AREA ({len(areas)} areas):")
            for area, area_entries in sorted(areas.items()):
                area_total = sum(float(e.get('total', 0)) for e in area_entries)
                text_parts.append(f"  {area}: ${area_total:,.2f} ({len(area_entries)} items)")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating estimate text: {e}")
            return f"Estimate data processing error: {str(e)}"
    
    def _create_flooring_estimate_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create text representation for flooring estimate"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            
            total_cost = sum(float(e.get('totalCost', 0)) for e in entries)
            total_sale = sum(float(e.get('salePrice', 0)) for e in entries)
            
            text_parts = [
                f"FLOORING ESTIMATE DATA",
                f"Job: {job_name}",
                f"Total Rows: {len(entries)}",
                f"Total Cost: ${total_cost:,.2f}",
                f"Total Sale Price: ${total_sale:,.2f}",
                f"Profit: ${total_sale - total_cost:,.2f}"
            ]
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating flooring estimate text: {e}")
            return f"Flooring estimate error: {str(e)}"
    
    def _create_schedule_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create text representation for schedule data"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            
            valid_tasks = [e for e in entries if e.get('task', '').strip()]
            total_hours = sum(float(e.get('hours', 0)) for e in valid_tasks)
            total_consumed = sum(float(e.get('consumed', 0)) for e in valid_tasks)
            
            text_parts = [
                f"SCHEDULE DATA",
                f"Job: {job_name}",
                f"Total Tasks: {len(valid_tasks)}",
                f"Total Planned Hours: {total_hours:,.1f}",
                f"Total Consumed Hours: {total_consumed:,.1f}"
            ]
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating schedule text: {e}")
            return f"Schedule data error: {str(e)}"
    
    def categorize_cost_code(self, cost_code: str) -> str:
        """Categorize cost codes"""
        if not cost_code:
            return "Unknown"
        
        code_lower = cost_code.lower()
        
        if 'subcontractor' in code_lower:
            return "Subcontractors"
        
        import re
        suffix_match = re.search(r'\d+([SMLO])\b', cost_code)
        if suffix_match:
            suffix = suffix_match.group(1).upper()
            if suffix == 'S':
                return "Subcontractors"
            elif suffix == 'M':
                return "Materials"
            elif suffix == 'L':
                return "Labor"
            elif suffix == 'O':
                return "Other/Overhead"
        
        if any(word in code_lower for word in ['material', 'materials']):
            return "Materials"
        elif any(word in code_lower for word in ['labor', 'labour']):
            return "Labor"
        elif any(word in code_lower for word in ['permit', 'fee', 'overhead']):
            return "Other/Overhead"
        
        return "Other"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def create_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for document"""
        data_type = job_data.get('data_type', 'consumed')
        
        base_metadata = {
            'job_name': str(job_data.get('job_name', 'Unknown')),
            'company_id': str(job_data.get('company_id', '')),
            'job_id': str(job_data.get('job_id', '')),
            'data_type': data_type,
            'granularity': 'job'
        }
        
        return base_metadata
    
    def create_estimate_row_metadata(self, row_data: Dict[str, Any], job_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for estimate row"""
        return {
            'job_name': str(job_context.get('job_name', 'Unknown')),
            'company_id': str(job_context.get('company_id', '')),
            'job_id': str(job_context.get('job_id', '')),
            'row_number': int(row_data.get('row_number', 0)),
            'document_type': 'estimate_row',
            'data_type': 'estimate',
            'granularity': 'row',
            'area': str(row_data.get('area', '')),
            'task_scope': str(row_data.get('taskScope', '')),
            'cost_code': str(row_data.get('costCode', '')),
            'description': str(row_data.get('description', ''))[:200],
            'total': float(row_data.get('total', 0)),
            'budgeted_total': float(row_data.get('budgetedTotal', 0))
        }
    
    def create_estimate_row_text(self, row_data: Dict[str, Any], job_context: Dict[str, Any]) -> str:
        """Create text for estimate row"""
        row_num = row_data.get('row_number', 0)
        area = row_data.get('area', 'General')
        task = row_data.get('taskScope', 'Unknown')
        cost_code = row_data.get('costCode', 'Unknown')
        description = row_data.get('description', '')
        total = float(row_data.get('total', 0))
        budgeted = float(row_data.get('budgetedTotal', 0))
        
        text = f"""ESTIMATE ROW #{row_num}
Job: {job_context.get('job_name', 'Unknown')}
Area: {area}
Task: {task}
Cost Code: {cost_code}
Description: {description}
Estimated: ${total:,.2f}
Budgeted: ${budgeted:,.2f}
Variance: ${budgeted - total:,.2f}"""
        
        return text
    
    def create_flooring_estimate_row_metadata(self, row_data: Dict[str, Any], job_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for flooring estimate row"""
        return {
            'job_name': str(job_context.get('job_name', 'Unknown')),
            'company_id': str(job_context.get('company_id', '')),
            'job_id': str(job_context.get('job_id', '')),
            'row_number': int(row_data.get('row_number', 0)),
            'document_type': 'flooring_estimate_row',
            'data_type': 'flooring_estimate',
            'granularity': 'row',
            'vendor': str(row_data.get('vendor', '')),
            'item_material_name': str(row_data.get('itemMaterialName', '')),
            'total_cost': float(row_data.get('totalCost', 0)),
            'sale_price': float(row_data.get('salePrice', 0))
        }
    
    def create_flooring_estimate_row_text(self, row_data: Dict[str, Any], job_context: Dict[str, Any]) -> str:
        """Create text for flooring estimate row"""
        row_num = row_data.get('row_number', 0)
        item = row_data.get('itemMaterialName', 'Unknown')
        vendor = row_data.get('vendor', 'Unknown')
        cost = float(row_data.get('totalCost', 0))
        sale = float(row_data.get('salePrice', 0))
        
        text = f"""FLOORING ESTIMATE ROW #{row_num}
Job: {job_context.get('job_name', 'Unknown')}
Item: {item}
Vendor: {vendor}
Cost: ${cost:,.2f}
Sale Price: ${sale:,.2f}
Profit: ${sale - cost:,.2f}"""
        
        return text