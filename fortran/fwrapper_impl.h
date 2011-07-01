/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2011 by Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/**
 * @file fwrapper_impl.h
 * @brief C++から呼び出されるユーザーアプリケーション内サブルーチンの宣言
 */

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

  /**
   * @brief アプリケーションの初期化処理が実装される。
   * 計算実行前に1度だけコールされる。
   * 配列の確保やパラメータの取り出し等、初期化処理を行う。
   *
   * @param[in] caller 呼び出し元ワーカのポインタ
   */
  void alps_init_(const int* caller);

  /**
   * @brief alps::ObservableSetの初期化が実装される。
   * 
   * @param[in] caller 呼び出し元ワーカのポインタ
   */
  void alps_init_observables_(const int* caller);

  /**
   * @brief アプリケーションの計算ロジックが実装される。
   * 
   * alps_progress_ が1以上を返すまでALPSから繰り返しコールされる。
   *
   * @param[in] caller 呼び出し元ワーカのポインタ
   */
  void alps_run_(const int* caller);

  /**
   * @brief 計算の進捗状態を返す。
   * 
   * 1以上の値を返すと計算ループ(alps_run_ の呼び出し)が終了する。
   *
   * @param[in] caller 呼び出し元ワーカのポインタ
   * @param[out] progress 進捗状態
   * @retval 1未満 計算中
   * @retval 1以上 計算終了
   */
  void alps_progress_(double* progress, const int* caller);

  /**
   * @brief サーマライズ状態を返す。
   * 
   * 1を返すとサーマライズが完了したとみなされ、 alps::ObservableSet への結果保存が開始される。
   *
   * @param[in] caller 呼び出し元ワーカのポインタ
   * @param[out] is_thermalized サーマライズ状態
   * @retval 0 サーマライズ未完了
   * @retval 1 サーマライズ完了
   */
  void alps_is_thermalized_(int* is_thermalized, const int* caller);

  /**
   * @brief アプリケーションの終了処理が実装される
   * 
   * @param[in] caller 呼び出し元ワーカのポインタ
   */
  void alps_finalize_(const int* caller);

  /**
   * @brief リスタートファイルの出力処理が実装される。
   * @param[in] caller 呼び出し元ワーカのポインタ
   */
  void alps_save_(const int* caller);

  /**
   * @brief リスタートファイルの入力処理が実装される。
   * @param[in] caller 呼び出し元ワーカのポインタ
   */
  void alps_load_(const int* caller);

#ifdef __cplusplus
}
#endif /* __cplusplus */

