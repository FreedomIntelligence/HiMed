import os
import argparse
from pdf2image import convert_from_path

def pdf_to_images_for_folder(input_dir: str, output_dir: str, dpi: int=300):
    if not os.path.isdir(input_dir):
        raise ValueError(f'输入文件夹不存在: {input_dir}')
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(input_dir, filename)
        pdf_name = os.path.splitext(filename)[0]
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)
        print(f'[INFO] 正在处理: {pdf_path}')
        try:
            pages = convert_from_path(pdf_path, dpi=dpi)
            for idx, page in enumerate(pages, start=1):
                img_name = f'{idx}.png'
                img_path = os.path.join(pdf_output_dir, img_name)
                page.save(img_path, 'PNG')
                print(f'  -> 已保存: {img_path}')
        except Exception as e:
            print(f'[ERROR] 处理 {filename} 时出错: {e}')

def main():
    parser = argparse.ArgumentParser(description='将指定文件夹下的所有 PDF 文件拆成逐页 PNG')
    parser.add_argument('--input-dir', type=str, required=True, help='PDF 输入文件夹路径')
    parser.add_argument('--output-dir', type=str, required=True, help='图片输出文件夹路径')
    parser.add_argument('--dpi', type=int, default=400, help='图片清晰度，调高更清晰（建议 300–600），默认400')
    args = parser.parse_args()
    pdf_to_images_for_folder(args.input_dir, args.output_dir, args.dpi)
if __name__ == '__main__':
    main()
