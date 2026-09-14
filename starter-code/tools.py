import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# TODO: Hoàn thiện hàm này — đọc file product_catalog.json, lọc theo category và max_price.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    try:
        with open(catalog_file, "r", encoding="utf-8") as file:
            products = json.load(file)
    except (OSError, json.JSONDecodeError):
        return [{"error": "Product catalog could not be read."}]

    normalized_category = category.strip().lower()
    try:
        price_limit = int(max_price)
    except (TypeError, ValueError):
        return [{"error": "max_price must be an integer."}]

    return [
        product
        for product in products
        if product.get("category", "").lower() == normalized_category
        and product.get("price_vnd", float("inf")) <= price_limit
    ]


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# TODO: Hoàn thiện hàm này — tạo ticket mới và lưu vào support_tickets.json.
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        try:
            with open(tickets_file, "r", encoding="utf-8") as file:
                loaded_tickets = json.load(file)
            if isinstance(loaded_tickets, list):
                existing_tickets = loaded_tickets
        except (OSError, json.JSONDecodeError):
            # A new ticket can still be recorded when an empty/corrupt mock file
            # is encountered, rather than crashing the conversation.
            existing_tickets = []

    normalized_priority = str(priority).lower().strip()
    if normalized_priority not in {"low", "medium", "high"}:
        normalized_priority = "medium"

    now = datetime.now()
    ticket_id = f"TK-{now.strftime('%Y%m%d')}-{len(existing_tickets) + 1:03d}"
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name.strip(),
        "issue_description": issue_description.strip(),
        "priority": normalized_priority,
        "status": "open",
        "created_at": now.isoformat() + "+07:00",
        "category": "general",
    }
    existing_tickets.append(new_ticket)

    with open(tickets_file, "w", encoding="utf-8") as file:
        json.dump(existing_tickets, file, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": new_ticket["customer_name"],
        "priority": normalized_priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# TODO: Định nghĩa JSON Schema cho từng tool (name, description, parameters).
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm hoặc dịch vụ Vingroup theo danh mục và giá tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm cần tra cứu.",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": "Ngân sách tối đa, tính bằng VNĐ.",
                    "minimum": 0,
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": "Tạo ticket hỗ trợ cho một vấn đề của khách hàng.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ và tên khách hàng.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả cụ thể vấn đề cần hỗ trợ.",
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên của ticket.",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
