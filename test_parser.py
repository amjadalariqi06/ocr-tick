import unittest
from main import parse_universal_document


class ParserRegressionTests(unittest.TestCase):
    def test_visa_fields_english(self):
        result = parse_universal_document(
            "المملكة العربية السعودية\nName: AHMED ALI\n"
            "Nationality: YEMEN\nUmrah Operator: Al Manasik Co\n"
            "Passport No: A1234567\nVisa No: 61234567890\nIssue Date: 12/10/2026"
        )
        self.assertEqual(result["full_name"], "AHMED ALI")
        self.assertEqual(result["passport_number"], "A1234567")
        self.assertEqual(result["reference_number"], "61234567890")
        self.assertEqual(result["date"], "12/10/2026")
        self.assertEqual(result["nationality"], "اليمن")
        self.assertEqual(result["extra1"], "Al Manasik Co")

    def test_reversed_arabic_name_and_alsam_noise(self):
        # Simulates reversed Arabic OCR output with "السم" attached
        result = parse_universal_document(
            "المملكة العربية السعودية\n"
            "الاسم: العريقي قايد أحمد أمجد السم\n"
            "شركة العمرة: المحدودة العمرة و الحج لخدمات المشاعر شركة\n"
            "الجنسية: اليمنية الجمهورية\n"
            "رقم الجواز: 09876543\nرقم التأشيرة: 61122334455"
        )
        self.assertEqual(result["full_name"], "أمجد أحمد قايد العريقي")
        self.assertEqual(result["extra1"], "شركة المشاعر لخدمات الحج و العمرة المحدودة")
        self.assertEqual(result["nationality"], "اليمن")

    def test_yemenia_ticket(self):
        result = parse_universal_document(
            "Yemenia Airways\nPASSENGER NAME: AHMED ALI\n"
            "TICKET NUMBER: 6351234567890\n12.OCT.2026"
        )
        self.assertEqual(result["full_name"], "AHMED ALI")
        self.assertEqual(result["reference_number"], "6351234567890")
        self.assertEqual(result["document_type"], "تذكرة طيران دولية")

    def test_arabic_digits(self):
        result = parse_universal_document("رقم التأشيرة: ٦١٢٣٤٥٦٧٨٩٠")
        self.assertEqual(result["reference_number"], "61234567890")


if __name__ == "__main__":
    unittest.main()
