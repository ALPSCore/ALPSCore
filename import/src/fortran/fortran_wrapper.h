/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
 * @file fortran_wrapper.h
 * @brief alps::fortran_wrapper クラス宣言
 */
#include <alps/parapack/worker.h>
#include "fwrapper_impl.h"

namespace alps
{
  /**
   * @brief ALPS Worker
   *
   * ALPSからの呼び出しに応じてFortranコードを呼び出す
   */
  class fortran_wrapper : public alps::parapack::abstract_worker
  {
  public:
    /**
     * @brief fortran_wrapper クラスのポインタを保持する構造体。
     *
     * Fortranとのインターフェースのため、ポインタとint配列の共用体として管理する。
     */
    struct alps_fortran_ptr
    {
      union
      {
	fortran_wrapper* m_pointer; //!< ポインタ
	int m_integer[2]; //!< int配列
      };
      
      /**
       * @brief コンストラクタ
       *
       * 32bit環境で実行された場合、 ::m_integer[1] は0になる
       * @param[in] ptr fortran_wrapper へのポインタ
       */
      alps_fortran_ptr(fortran_wrapper* ptr)
      {
	m_integer[0] = m_integer[1] = 0;
	m_pointer = ptr;
      }

      /**
       * @brief コンストラクタ
       * @param[in] ptr fortran_wrapper へのポインタが格納されたint配列
       */
      alps_fortran_ptr(int* ptr)
      {
	m_integer[0] = ptr[0];
	m_integer[1] = ptr[1];
      }
    };

  private:
    const alps::Parameters* m_parameters; //!< パラメータコンテナ
    alps::ObservableSet* m_observable; //!< 結果コンテナ
    mutable alps::ODump* m_odump; //!< リスタートファイル出力用ストリーム
    alps::IDump* m_idump; //!< リスタートファイル入力用ストリーム
    alps_fortran_ptr m_self; //!< 自身へのポインタ

  public:
    /**
     * @brief コンストラクタ
     *
     * ユーザーアプリケーションのalps_init_サブルーチンを呼び出す。
     * @param[in] params パラメータコンテナ
     */
    fortran_wrapper(alps::Parameters const& params) : m_self(this), m_odump(NULL), m_idump(NULL), m_observable(NULL)
      {
	m_parameters = &params;
	alps_init_(m_self.m_integer);
      }

      /**
       * @brief デストラクタ
       *
       * ユーザーアプリケーションのalps_init_サブルーチンを呼び出す。
       */
      virtual ~fortran_wrapper()
	{
	  alps_finalize_(m_self.m_integer);
	}

      /**
       * @brief ObservableSetの初期化
       *
       * ユーザーアプリケーションのalps_init_observables_を呼び出す。
       * @param[in] params パラメータコンテナ
       * @param[out] ob 計算結果コンテナ
       */
      void init_observables(alps::Parameters const& params, alps::ObservableSet& ob)
      {
	m_observable = &ob;
	alps_init_observables_(m_self.m_integer);
      }

      /**
       * @brief サーマライズ状態を返す
       *
       * ユーザーアプリケーションのalps_is_thermalized_サブルーチンを呼び出す。
       * @retval true サーマライズ完了
       * @retval flase サーマライズ未完了
       */
      bool is_thermalized() const
      {
	int retval;
	alps_is_thermalized_(&retval, m_self.m_integer);
	return retval == 0 ? false : true;
      }

      /**
       * @brief 実行ステータスを返す
       *
       * ユーザーアプリケーションのalps_progress_サブルーチンを呼び出す。
       * @retval 1以上：完了
       * @retval 1未満：実行中
       */
      double progress() const
      {
	double retval;
	alps_progress_(&retval, m_self.m_integer);
	return retval;
      }

      /**
       * @brief 繰り返し計算処理
       *
       * ユーザーアプリケーションのalps_run_サブルーチンを呼び出す。
       * @param[out] obs 計算結果コンテナ
       */
      void run(alps::ObservableSet& obs)
      {
	m_observable = &obs;
	alps_run_(m_self.m_integer);
      }

      /**
       * @brief リスタートファイルを出力する
       *
       * ユーザーアプリケーションのalps_save_サブルーチンを呼び出す。
       * @param[out] odump 出力ストリーム
       */
      void save(alps::ODump& odump) const
      {
	m_odump = &odump;
	alps_save_(m_self.m_integer);
      }

      /**
       * @brief リスタートファイルを読み込む
       *
       * ユーザーアプリケーションのalps_load_サブルーチンを呼び出す。
       * @param[in] idump 入力ストリーム
       */
      void load(alps::IDump& idump)
      {
	m_idump = &idump;
	alps_load_(m_self.m_integer);
      }

      /**
       * @brief 自身に関連付けられたパラメータコンテナを返す。
       * @return パラメータコンテナ
       */
      const alps::Parameters* parameters()const
	{
	  return m_parameters;
	}

      /**
       * @brief 自身に関連づけられた出力ストリームを返す。
       * @return 出力ストリーム
       */
      alps::ODump& odump()
	{
	  return *m_odump;
	}

      /**
       * @brief 自身に関連づけられた入力ストリームを返す。
       * @return 入力ストリーム
       */
      alps::IDump& idump()
	{
	  return *m_idump;
	}

      /**
       * @brief 自身に関連付けられた計算結果コンテナを返す。
       * @return 計算結果コンテナ
       */
      alps::ObservableSet& observables()const
	{
	  return *m_observable;
	}
  };
}
