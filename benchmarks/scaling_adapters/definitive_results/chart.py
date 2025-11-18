import os
import re
import json
import glob
import argparse
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def extract_results_model(path: str, rank: int) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def extract_experiment_metric(path: str, m: int, rank: int, rate: float) -> Dict[str, float]:
        output: Dict[str, float] = {}
        filenames: List[str] = glob.glob(os.path.join(path, 'slora-infqps-*_intermediate.json'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as metrics_file:
            metrics: dict = json.load(metrics_file)

        # compute full throughput (input + output)
        output['throughput'] = float(metrics['input_throughput']) + float(metrics['output_throughput'])

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'rank', 'rate']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            m: int = int(folder.replace('__', '').split('_')[0])
            rate: float = float(folder.replace('__', '').split('_')[1])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder), m, rank, rate)
            except Exception as e:
                print(e)
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
                try:
                    with open(os.path.join(path, folder, 'server_err.log')) as f:
                        error_log: str = f.read()
                        if 'ValueError: No available memory for the cache blocks' in error_log:
                            error_message += 'Not enough memory'
                        elif 'torch.cuda.OutOfMemoryError: CUDA out of memory' in error_log:
                            error_message += 'Not enough memory'
                        elif 'ValueError: The model\'s max seq len (4096) is larger than the maximum number of tokens that can be stored in KV cache' in error_log:
                            error_message += 'Not enough memory'
                        elif 'RuntimeError: CUDA error: uncorrectable ECC error encountered' in error_log:
                            error_message += 'ECC error'
                            rerun_errors.append(os.path.join(path, folder))
                        elif 'RuntimeError: CUDA error: an illegal memory access was encountered' in error_log:
                            error_message += 'Memory access error'
                            rerun_errors.append(os.path.join(path, folder))
                        elif '[Errno 98] error while attempting to bind on address' in error_log:
                            error_message += 'Port bind error'
                            rerun_errors.append(os.path.join(path, folder))
                        else:
                            error_message += 'Unknown error'
                            unknown_errors += 1
                            print(folder)
                except Exception as e:
                    error_message += 'Unknown error'
                    unknown_errors += 1
                    print(folder)
                # print(error_message)
                metrics = {}
            metrics['m'] = m
            metrics['rank'] = rank
            metrics['rate'] = rate
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return results


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str, filter_in: Tuple[str, str] = None, additional_line: str = None) -> List[
    Tuple[str, List[int], List[float]]]:
    output_tmp: Dict[str, Tuple[List[int], List[float]]] = {}
    for item in results:
        selection_id = item[selection]
        if filter_in is not None and str(item[filter_in[0]]) != filter_in[1]:
            continue
        if selection_id not in output_tmp:
            output_tmp[selection_id] = ([], [])
            if additional_line is not None:
                output_tmp[selection_id] = ([], [], [])
        if x_axis not in item:
            output_tmp[selection_id][0].append(None)
        else:
            output_tmp[selection_id][0].append(item[x_axis])
        if y_axis not in item:
            output_tmp[selection_id][1].append(None)
        else:
            output_tmp[selection_id][1].append(item[y_axis])
        if additional_line is not None:
            if additional_line not in item:
                output_tmp[selection_id][2].append(None)
            else:
                output_tmp[selection_id][2].append(item[additional_line])
    output: List[Tuple[str, List[int], List[float]]] = []
    for key, values in output_tmp.items():
        if additional_line is None:
            x_values, y_values = values
        else:
            x_values, y_values, z_values = values
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        if additional_line is not None:
            z_line = [z_value for index, z_value in enumerate(z_values) if
                  z_value is not None and x_values[index] is not None]
        y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        if additional_line is not None:
            z_line = [z_value for _, z_value in sorted(zip(x_line, z_line))]
        x_line.sort()
        if additional_line is None:
            output.append(
                (
                    key,
                    x_line,
                    y_line
                )
            )
        else:
            output.append(
                (
                    key,
                    x_line,
                    y_line,
                    z_line
                )
            )
    output = [value for _, value in sorted(zip([value[0] for value in output], output))]

    return output


def plot_single_rank_gpu_capacity(
        rank_results: List[Dict[str, float]],
        rank: int,
        output_path: str,
        title: str,
) -> None:
    """
    Plot GPU capacity for a single rank
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (7, 5)  # Consistent size for single plot
    })

    # Prepare results for the rank
    rank_throughput = __prepare_lines(
        rank_results,
        'm',
        'throughput',
        'rate'
    )

    # Create figure
    fig, ax = plt.subplots()
    
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    input_length = 250   # Medium request input length
    output_length = 231  # Medium request output length
    total_length = input_length + output_length

    # Plot the rank data
    lines = []
    for rate, x_line, y_line in rank_throughput:
        line = ax.plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'{rate}'
        )[0]
        lines.append(line)
        
        # Find and mark the point BEFORE theoretical throughput exceeds real by 10%
        x_line_array = np.asarray(x_line)
        y_line_array = np.asarray(y_line)
        y_line_ideal = x_line_array * rate * total_length
        
        for i in range(len(x_line)):
            if y_line_ideal[i] > y_line_array[i] * 1.1:
                # Mark the previous point with a big X (if it exists)
                if i > 0:
                    ax.plot(
                        x_line[i-1],
                        y_line[i-1],
                        marker='x',
                        markersize=12,
                        color=line.get_color(),
                        markeredgewidth=3
                    )
                break  # Only mark the first crossing point

    for index, (rate, x_line, y_line) in enumerate(rank_throughput):
        x_line_ideal = np.asarray(x_line)
        y_line_ideal = x_line_ideal * rate * total_length
        last_index = 0
        while last_index < len(x_line_ideal) and y_line_ideal[last_index] < (y_line[last_index] * 1.5):
            last_index += 1
        ax.plot(
            x_line_ideal[:(last_index + 1)],
            y_line_ideal[:(last_index + 1)],
            marker='',
            color=lines[index].get_color(),
            linestyle='dotted'
        )

    ax.set_ylabel('total throughput (tokens/s)')
    ax.set_xlabel('adapters (#)')
    ax.legend(loc='upper right', title='rates')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'gpu_capacity_{title}_rank_{rank}.pdf'), format='pdf', bbox_inches='tight', dpi=400)
    plt.close('all')
    print(f"Plot saved to {os.path.join(output_path, f'gpu_capacity_{title}_rank_{rank}')}")


def plot_gpu_capacity(
        rank_8_results: List[Dict[str, float]],
        rank_32_results: List[Dict[str, float]],
        output_path: str,
        title: str,
) -> None:
    # Prepare results for both ranks
    rank_8_throughput = __prepare_lines(
        rank_8_results,
        'm',
        'throughput',
        'rate'
    )
    
    rank_32_throughput = __prepare_lines(
        rank_32_results,
        'm',
        'throughput',
        'rate'
    )

    # Create figure with 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    input_length = 250   # Medium request input length
    output_length = 231  # Medium request output length
    total_length = input_length + output_length

    # Plot rank 8
    lines = []
    for rate, x_line, y_line in rank_8_throughput:
        line = axs[0].plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'{rate}'
        )[0]
        lines.append(line)
        
        # Find and mark the point BEFORE theoretical throughput exceeds real by 10%
        x_line_array = np.asarray(x_line)
        y_line_array = np.asarray(y_line)
        y_line_ideal = x_line_array * rate * total_length
        
        for i in range(len(x_line)):
            if y_line_ideal[i] > y_line_array[i] * 1.1:
                # Mark the previous point with a big X (if it exists)
                if i > 0:
                    axs[0].plot(
                        x_line[i-1],
                        y_line[i-1],
                        marker='x',
                        markersize=12,
                        color=line.get_color(),
                        markeredgewidth=3
                    )
                break  # Only mark the first crossing point

    for index, (rate, x_line, y_line) in enumerate(rank_8_throughput):
        x_line_ideal = np.asarray(x_line)
        y_line_ideal = x_line_ideal * rate * total_length
        last_index = 0
        while last_index < len(x_line_ideal) and y_line_ideal[last_index] < (y_line[last_index] * 1.5):
            last_index += 1
        axs[0].plot(
            x_line_ideal[:(last_index + 1)],
            y_line_ideal[:(last_index + 1)],
            marker='',
            color=lines[index].get_color(),
            linestyle='dotted'
            # Removed label parameter to hide from legend
        )

    axs[0].set_ylabel('throughput (tokens/s)', fontsize=10)
    axs[0].set_xlabel('loaded and served adapters (#)', fontsize=10)
    axs[0].set_title(f'MediumRequest rank 8')

    # Plot rank 32
    lines = []
    for rate, x_line, y_line in rank_32_throughput:
        line = axs[1].plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'{rate}'
        )[0]
        lines.append(line)
        
        # Find and mark the point BEFORE theoretical throughput exceeds real by 10%
        x_line_array = np.asarray(x_line)
        y_line_array = np.asarray(y_line)
        y_line_ideal = x_line_array * rate * total_length
        
        for i in range(len(x_line)):
            if y_line_ideal[i] > y_line_array[i] * 1.1:
                # Mark the previous point with a big X (if it exists)
                if i > 0:
                    axs[1].plot(
                        x_line[i-1],
                        y_line[i-1],
                        marker='x',
                        markersize=12,
                        color=line.get_color(),
                        markeredgewidth=3
                    )
                break  # Only mark the first crossing point

    for index, (rate, x_line, y_line) in enumerate(rank_32_throughput):
        x_line_ideal = np.asarray(x_line)
        y_line_ideal = x_line_ideal * rate * total_length
        last_index = 0
        while last_index < len(x_line_ideal) and y_line_ideal[last_index] < (y_line[last_index] * 1.5):
            last_index += 1
        axs[1].plot(
            x_line_ideal[:(last_index + 1)],
            y_line_ideal[:(last_index + 1)],
            marker='',
            color=lines[index].get_color(),
            linestyle='dotted'
            # Removed label parameter to hide from legend
        )

    axs[1].set_ylabel('throughput (tokens/s)', fontsize=10)
    axs[1].set_xlabel('loaded and served adapters (#)', fontsize=10)

    # Add legends
    for ax in axs:
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'gpu_capacity_{title}'), bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved to {os.path.join(output_path, f'gpu_capacity_{title}')}")


def main():
    parser = argparse.ArgumentParser(description='Plot GPU capacity for medium requests')
    parser.add_argument('--data-path', type=str, required=True, 
                       help='Path to the mean_dataset directory containing rank subdirectories')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save the output plot')
    parser.add_argument('--rank', type=int, help='If specified, only plot data for this rank')
    args = parser.parse_args()

    if args.rank is not None:
        # Single rank mode
        rank_path = os.path.join(args.data_path, f'rank_{args.rank}')
        if not os.path.exists(rank_path):
            print(f"Error: Could not find rank_{args.rank} directory in {args.data_path}")
            return
            
        results_rank = extract_results_model(rank_path, args.rank)
        
        plot_single_rank_gpu_capacity(
            results_rank,
            args.rank,
            args.output_path,
            'medium_requests_total_throughput'
        )
    else:
        # Dual rank mode (original behavior)
        results_mean_rank_8 = extract_results_model(os.path.join(args.data_path, 'rank_8'), 8)
        results_mean_rank_32 = extract_results_model(os.path.join(args.data_path, 'rank_32'), 32)

        plot_gpu_capacity(
            results_mean_rank_8,
            results_mean_rank_32,
            args.output_path,
            'medium_requests_total_throughput'
        )


if __name__ == '__main__':
    main()
